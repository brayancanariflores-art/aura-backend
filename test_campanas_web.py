import base64
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ajustes_marca import AjusteNoAplicado
from campanas_web import crear_campana, generar_toma, logo_guardado, motor_para_pedido, pide_campana, validar_movimiento
from control_campana import contiene_accion, revisar_identidad
from movimiento_marca import recortar_logo
from registro_superficie import Registro, registros_del_clip


class CampanaWebTest(unittest.TestCase):
    def test_pedido_corto_y_marca_correcta(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"AURA_MARCAS_DIR":tmp}, clear=True):
            carpeta=Path(tmp)
            Image.new("RGBA",(48,48),"green").save(carpeta/"hatun-runa.png")
            self.assertTrue(pide_campana("Reel profesional de publicidad con el logo Hatun Runa"))
            self.assertFalse(pide_campana("video narrado de mi receta"))
            self.assertEqual(logo_guardado("Para Hatun Runa").name,"hatun-runa.png")
            self.assertIsNone(logo_guardado("Hatun Runaway"))
            with self.assertRaisesRegex(AjusteNoAplicado,"activad"):
                crear_campana(carpeta,"Crea un reel de Hatun Runa con personajes fantásticos",None,None,"vertical",None)

    def test_una_campana_no_se_sustituye_por_la_plantilla(self):
        orden = "Créame un reel profesional de publicidad con el logo Hatun Runa"
        self.assertEqual(motor_para_pedido(orden, {}), "generativo")
        self.assertEqual(motor_para_pedido("Haz una animación de marca", {}), "animacion")
        self.assertEqual(motor_para_pedido("Usa cobre", {"tipo": "animacion"}), "animacion")
        self.assertEqual(motor_para_pedido("Ahora crea escenas originales", {"tipo": "animacion"}), "generativo")
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch("campanas_web.renderizar") as plantilla:
            carpeta = Path(tmp)
            Image.new("RGBA", (24, 24), "green").save(carpeta/"logo-campana.png")
            with self.assertRaisesRegex(AjusteNoAplicado, "generar escenas originales"):
                crear_campana(carpeta, orden, None, None, "vertical", None)
            plantilla.assert_not_called()
            self.assertFalse((carpeta/"campana.mp4").exists())

    def test_conserva_blancos_internos_del_logo(self):
        with TemporaryDirectory() as tmp:
            ruta=Path(tmp)/"arte.png"
            a=np.full((30,30,3),255,dtype=np.uint8)
            a[5:25,5:25]=0
            a[8:22,8:22]=255
            Image.fromarray(a).save(ruta)
            arte=recortar_logo(ruta)
            self.assertEqual(arte.size,(20,20))
            self.assertEqual(arte.getpixel((9,9)),(255,255,255,255))

    def test_rechaza_movimiento_congelado(self):
        with patch("campanas_web.ejecutar",return_value=bytes(48*48*8)):
            with self.assertRaisesRegex(AjusteNoAplicado,"muy poco movimiento"):
                validar_movimiento(None)

    def test_paneo_y_cambio_de_luz_no_son_accion_del_sujeto(self):
        rng = np.random.default_rng(24)
        cuadro = rng.integers(10, 185, (80, 72), dtype=np.uint8)
        paneo = np.roll(cuadro, 1, axis=1) + 5
        self.assertFalse(contiene_accion(cuadro, paneo))
        movimiento = cuadro.copy()
        movimiento[15:60, 15:45] = np.roll(cuadro[15:60, 15:45], 3, axis=0)
        self.assertTrue(contiene_accion(cuadro, movimiento))

    def test_verifica_logo_y_rubro_al_abrir_el_mp4(self):
        with TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "logo.png"
            im = Image.new("RGB", (32, 32), "green")
            im.save(ruta)
            cliente = Mock()
            servicio = cliente.with_options.return_value.chat.completions.create
            respuesta = Mock()
            respuesta.choices = [Mock()]
            respuesta.choices[0].message.content = '{"tema_correcto":false,"logo_correcto":false}'
            servicio.return_value = respuesta
            with patch("control_campana.inspeccionar", return_value=({}, [im]*4)):
                with self.assertRaisesRegex(AjusteNoAplicado, "identidad visual"):
                    revisar_identidad(ruta, ruta, "Hatun Runa Resto Bar", cliente, "vision")
            contenido = servicio.call_args.kwargs["messages"][1]["content"]
            self.assertEqual(sum(item["type"]=="image_url" for item in contenido), 5)

    def test_contrato_motor_y_continuidad_de_revision(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ,{"GEMINI_API_KEY":"mock"}), patch("campanas_web.httpx.post") as post, patch("campanas_web.revisar_accion") as comprobar_accion, patch("campanas_web.inspeccionar",return_value=({"duracion":8,"ancho":720,"alto":1280},[])), patch("campanas_web.ejecutar") as ff:
            p=Path(tmp)
            post.return_value.json.return_value={"id":"mock-id","steps":[{"type":"model_output","content":[{"type":"video","mime_type":"video/mp4","data":base64.b64encode(b"fixture").decode()}]}]}
            _,historia=generar_toma(p,{"escena":"Movimiento continuo"},"vertical",None,True,"anterior")
            payload=post.call_args.kwargs["json"]
            self.assertEqual(payload["previous_interaction_id"],"anterior")
            self.assertEqual(payload["response_format"]["aspect_ratio"],"9:16")
            self.assertEqual(historia,"mock-id")
            comprobar_accion.assert_called_once_with(p/"toma.mp4")
            self.assertNotIn("0:a:0?",ff.call_args.args[0])

    def test_reutiliza_medicion_y_descarta_la_de_otro_clip(self):
        with TemporaryDirectory() as tmp, patch("registro_superficie.preparar_registros") as medir:
            clip=Path(tmp)/"original.mp4"
            clip.write_bytes(b"primer original")
            reg=Registro(np.array([0.,1.]),np.zeros((2,10)),np.arange(5),None)
            medir.return_value=[reg,None]
            grupos=[([[0,0,0,20,20]],1),([[1,0,0,20,20]],2)]
            registros_del_clip(clip,grupos,(0,0,50,50))
            segunda=registros_del_clip(clip,grupos,(0,0,50,50))
            self.assertEqual(medir.call_count,1)
            np.testing.assert_equal(segunda[0].estados,reg.estados)
            self.assertIsNone(segunda[0].mascara)
            self.assertIsNone(segunda[1])
            clip.write_bytes(b"otro original")
            registros_del_clip(clip,grupos,(0,0,50,50))
            self.assertEqual(medir.call_count,2)
