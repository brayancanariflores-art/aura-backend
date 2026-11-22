import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from tamano_marca import escala_marca, solo_tamano
from tinta_video import acabado_impreso, coordenadas_arte, imprimir, preparar_tinta


class TintaVideoTest(unittest.TestCase):
    def test_ajuste_de_acabado(self):
        self.assertTrue(acabado_impreso("Cambia la pegatina.\nAjuste solicitado: como marca de agua en el vaso"))
        self.assertFalse(acabado_impreso("Marca de agua.\nAjuste solicitado: ahora haz un sticker"))
        self.assertFalse(acabado_impreso("Reemplaza la etiqueta de la caja"))

    def test_escala_en_ordenes_y_revisiones(self):
        self.assertEqual(escala_marca("Agrándalo a un treinta o treinta y cinco por ciento"), 1.35)
        self.assertEqual(escala_marca("Haz el logo un 30 % más grande"), 1.3)
        self.assertEqual(escala_marca("Aumenta el volumen un 35 %"), 1)
        self.assertEqual(escala_marca("Reduce la opacidad del logo un 35 %"), 1)
        previo = "Agranda el logo un 12 %"
        self.assertEqual(escala_marca(previo + "\nAjuste solicitado: Agranda el logo un 35 %"), 1.35)
        self.assertEqual(escala_marca(previo + "\nAjuste solicitado: tamaño original"), 1)
        self.assertEqual(escala_marca(previo + "\nAjuste solicitado: reduce el brillo un 20 %"), 1.12)
        self.assertEqual(escala_marca("Achica el logo un 15 %"), .85)
        self.assertTrue(solo_tamano("Agranda el logo un treinta y cinco por ciento."))
        self.assertFalse(solo_tamano("Agranda el logo un 35 % y muévelo a la derecha"))

    def test_el_tamano_cambia_sin_mover_el_centro_ni_el_borrado(self):
        foto = Image.new("RGB", (190, 170), (175, 120, 60))
        ImageDraw.Draw(foto).rectangle((71, 57, 121, 113), fill=(244, 50, 78))
        tinta = np.zeros((60, 60, 4), np.float32)
        tinta[5:55, 5:55, 3] = 1
        def render(escala, arte):
            cuadro = foto.copy()
            sello, origen = imprimir(cuadro, (61, 50, 70, 70), arte, (1, 1), False, escala=escala)
            cuadro.paste(sello, origen, sello)
            return np.asarray(cuadro)
        def limites(factor):
            filas, columnas = np.nonzero(render(factor, tinta)[:, :, 0] < 70)
            return np.array([columnas.min(), filas.min()]), np.array([columnas.max(), filas.max()])
        a, b = limites(1)
        c, d = limites(1.35)
        np.testing.assert_allclose(d-c+1, 1.35*(b-a+1), atol=2)
        np.testing.assert_allclose((a+b)/2, (c+d)/2, atol=.5)
        np.testing.assert_array_equal(render(1, tinta*0), render(1.35, tinta*0))

    def test_el_papel_claro_deja_ver_el_vidrio(self):
        with TemporaryDirectory() as carpeta:
            ruta = Path(carpeta)/"logo.jpg"
            arte = Image.new("RGB",(90,90),"white")
            ImageDraw.Draw(arte).rectangle((30,25,60,68), fill=(10,40,110))
            arte.save(ruta,quality=95)
            tinta = preparar_tinta(ruta)
        self.assertLess(tinta[8,8,3], .02)
        self.assertGreater(tinta[45,45,3], .95)
        with TemporaryDirectory() as carpeta:
            ruta = Path(carpeta)/"logo.png"
            im = Image.new("RGBA",(30,30),(0,0,0,0))
            ImageDraw.Draw(im).rectangle((10,10,20,20),fill="white")
            im.save(ruta)
            self.assertEqual(preparar_tinta(ruta)[14,14,3],1)

    def test_al_girar_se_asoma_una_parte_del_arte(self):
        u = np.linspace(0,1,21)
        v = np.ones_like(u)*.5
        x,_,_ = coordenadas_arte(u,v,.15,(1,1),True,1)
        self.assertLess(x.max(),.5)
        x,_,_ = coordenadas_arte(u,v,.15,(1,-1),True,1)
        self.assertGreater(x.min(),.5)

    def test_quita_la_tinta_anterior_a_traves_de_las_partes_transparentes(self):
        color = (167,104,46)
        foto = Image.new("RGB",(200,140),color)
        ImageDraw.Draw(foto).rectangle((71,49,136,92),fill=(240,52,77))
        tinta = np.zeros((64,64,4),np.float32)
        tinta[24:40,24:40] = (.05,.12,.22,1.)
        parche, origen = imprimir(foto,(70,48,67,45),tinta,(1.5,1),True)
        foto.paste(parche,origen,parche)
        self.assertLess(np.abs(np.asarray(foto)[55,86].astype(float)-color).max(),18)
        self.assertLess(foto.getpixel((104,70))[0],115)
        self.assertEqual(foto.getpixel((50,45)),color)
