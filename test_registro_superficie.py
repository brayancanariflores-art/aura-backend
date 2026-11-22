import unittest

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from registro_superficie import IDENTIDAD, Registro, borde, proyectar, rastrear


class RegistroSuperficieTest(unittest.TestCase):
    def test_recupera_el_movimiento_de_la_tinta_que_la_caja_no_recoge(self):
        textura=ndimage.gaussian_filter(np.random.default_rng(24).normal(size=(90,125)),1.0).astype(np.float32)
        desplazamientos=[-9,-4,0,3,8,11]
        imagenes=[ndimage.shift(textura,(0,dx),order=1) for dx in desplazamientos]
        matrices,calidad=rastrear(imagenes,[[14,12,107,78]]*len(imagenes),2)
        for dx,matriz,n in zip(desplazamientos,matrices,calidad):
            centro=np.array([60.,45.])
            diferencia=centro@matriz[:2]+matriz[2]-centro
            self.assertGreater(n,6)
            np.testing.assert_allclose(diferencia,[dx,0],atol=.37)

    def test_respeta_los_limites_del_registro_y_el_contorno(self):
        forma=[70.,50.,15.,0.]
        rect=(65.,25.,187.,112.,5)
        estados=np.tile(np.r_[forma,IDENTIDAD.ravel()],(2,1))
        registro=Registro([2.,2.1],estados,rect,np.ones((140,259),np.float32))
        self.assertIsNone(registro.en(1.8))
        estado=registro.en(2.05)
        _,_,visible,_=proyectar(estado,np.array([[70.,130.]]),np.array([[50.,50.]]),1)
        np.testing.assert_allclose(visible,[[1,0]])

    def test_admite_un_borde_de_vaso_y_descarta_un_fondo_liso(self):
        im=Image.new("RGB",(200,150),(35,42,49))
        self.assertIsNone(borde(np.array(im),[80,65,70,68],115))
        d=ImageDraw.Draw(im);d.rectangle((46,38,161,51),fill="ivory")
        forma=borde(np.array(im),[80,65,70,68],115)
        self.assertIsNotNone(forma)
        np.testing.assert_allclose(forma[:3],[103.5,57.5,38],atol=2)
