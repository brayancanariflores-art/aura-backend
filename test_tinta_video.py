import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from tinta_video import acabado_impreso, coordenadas_arte, imprimir, preparar_tinta


class TintaVideoTest(unittest.TestCase):
    def test_ajuste_de_acabado(self):
        self.assertTrue(acabado_impreso("Cambia la pegatina.\nAjuste solicitado: como marca de agua en el vaso"))
        self.assertFalse(acabado_impreso("Marca de agua.\nAjuste solicitado: ahora haz un sticker"))
        self.assertFalse(acabado_impreso("Reemplaza la etiqueta de la caja"))

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
