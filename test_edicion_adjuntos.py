import io
import unittest
from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image, ImageDraw

from adjuntos import Solucion, Soluciones, procesar
from ediciones_chat import cancelar, ejecutar_edicion


class EdicionAdjuntos(unittest.TestCase):
    def setUp(self):
        im = Image.new("RGB", (130, 120), "#f8f8f8")
        dibujo = ImageDraw.Draw(im)
        dibujo.ellipse((20, 10, 110, 100), fill="white", outline="black", width=4)
        muestra = io.BytesIO()
        im.save(muestra, "JPEG")
        self.datos = muestra.getvalue()

    def test_marca_y_revision_conservan_la_figura(self):
        salida = procesar("image/jpeg", self.datos, "Marca de agua", None, "test", "marca")
        marca = Image.open(io.BytesIO(salida.body))
        self.assertEqual(marca.size, (130, 120))
        self.assertEqual(marca.getpixel((0, 0))[3], 0)
        self.assertEqual(marca.getpixel((60, 60))[3], 163)
        revision = procesar("image/png", salida.body, "Hazla más visible", None, "test", "marca")
        alfa = Image.open(io.BytesIO(revision.body)).getchannel("A")
        self.assertEqual(alfa.getextrema(), (0, 209))

    def test_consulta_incluye_la_imagen(self):
        class Vision:
            def create(self, **datos):
                self.entrada = datos["input"][0]["content"]
                return SimpleNamespace(output_text="Un círculo.")
        vision = Vision()
        salida = procesar("image/jpeg", self.datos, "¿Qué ves?", SimpleNamespace(responses=vision), "test", "consultar")
        self.assertEqual(salida, {"respuesta": "Un círculo."})
        self.assertTrue(vision.entrada[0]["image_url"].startswith("data:image/png;base64,"))

    def test_ejercicios_entregan_cada_resultado_y_conservan_las_pistas(self):
        class Vision:
            def parse(self, **datos):
                self.consulta = datos
                return SimpleNamespace(output_parsed=Soluciones(ejercicios=[
                    Solucion(ejercicio="1", calculo="2x = 6", resultado_o_dato_faltante="x = 3", comprobacion="2 × 3 = 6"),
                    Solucion(ejercicio="2", calculo="", resultado_o_dato_faltante="Falta la longitud de la viga.", comprobacion="")]))
            def create(self, **datos):
                return SimpleNamespace(output_text="Identifica los apoyos primero.")
        vision = Vision()
        cliente = SimpleNamespace(responses=vision)
        solucion = procesar("image/jpeg", self.datos, "Ayúdame a resolver estos ejercicios", cliente, "test", "consultar")
        self.assertEqual(vision.consulta["input"][0]["content"][0]["detail"], "high")
        self.assertIn("Respuesta: x = 3", solucion["respuesta"])
        self.assertIn("Respuesta: Falta la longitud", solucion["respuesta"])
        pista = procesar("image/jpeg", self.datos, "Solo dame una pista para el ejercicio", cliente, "test", "consultar")
        self.assertEqual(pista["respuesta"], "Identifica los apoyos primero.")

    def test_cancelacion_incluso_si_llega_antes_del_trabajo(self):
        orden = uuid4()
        self.assertEqual(cancelar(orden)["estado"], "cancelada")
        with self.assertRaises(HTTPException) as fallo:
            ejecutar_edicion(orden, "foto", self.datos)
        self.assertEqual(fallo.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
