import unittest
from pathlib import Path
from unittest.mock import MagicMock
from tempfile import TemporaryDirectory
import json

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from ajustes_marca import desplazamiento_explicito,planificar,pide_posicion
from texto_logo import cambio_de_leyenda, preparar_leyenda
from tinta_video import imprimir, muestra_de_textura


class AjustesMarcaTest(unittest.TestCase):
    def test_desplaza_los_dos_ejes_y_acumula_revision(self):
        orden='Baja el logo unos 60 píxeles y muévelo 20 píxeles hacia la izquierda.'
        self.assertTrue(pide_posicion(orden))
        self.assertEqual(desplazamiento_explicito(orden),(-20,60))
        grupos=[([[0,10,20,100,80],[1,20,20,100,80]],2)]
        previo={'posicion':[.1,-.25],'limpiar':True}
        res=planificar(None,grupos,(400,480),orden,previo,None,None)
        np.testing.assert_allclose(res['posicion'],[-.1,.5])
        self.assertTrue(res['limpiar'])

    def test_el_borrado_no_sigue_al_logo(self):
        im=Image.new('RGB',(220,220),(176,112,57))
        ImageDraw.Draw(im).rectangle((75,70,123,108),fill=(230,32,68))
        art=np.zeros((80,80,4),'float32')
        art[20:60,20:60,3]=1
        def render(offset):
            capa,xy=imprimir(im,(70,60,60,58),art,(1,1),False,posicion=offset,limpiar=True)
            out=im.copy();out.paste(capa,xy,capa);return np.asarray(out)
        original=render((0,0));movido=render((0,.7))
        a=np.argwhere(original[:,:,0]<80).mean(0);b=np.argwhere(movido[:,:,0]<80).mean(0)
        self.assertGreater(b[0]-a[0],33)
        np.testing.assert_allclose(movido[83,93].astype(float),[176,112,57],atol=5)

    def test_edita_una_leyenda_en_el_arte_y_verifica_el_resultado(self):
        self.assertEqual(cambio_de_leyenda('Reemplaza la palabra licorería por resto bar.'),('licorería','resto bar'))
        self.assertIsNone(cambio_de_leyenda('Reemplaza la marca por el logo adjunto'))
        with TemporaryDirectory() as directorio:
            ruta=Path(directorio)
            im=Image.new('RGBA',(170,130),'white');d=ImageDraw.Draw(im)
            fuente=ImageFont.load_default(size=18)
            d.text((40,20),'OLD',fill='black',font=fuente)
            d.rectangle((18,74,155,105),fill='blue')
            im.save(ruta/'marca.png')
            c=MagicMock();api=c.with_options.return_value.chat.completions.create
            def ans(valor):
                r=MagicMock();r.choices[0].message.content=json.dumps(valor);return r
            api.side_effect=[ans({'estado':'origen','mayusculas':True,'caja':[200,110,500,390]}),ans({'estado':'destino'})]
            out=preparar_leyenda(ruta,'Reemplaza «OLD» por «NEW»',c,'modelo')
            with Image.open(out) as mod:
                a,b=np.array(im),np.array(mod)
            self.assertTrue(np.any(a[:52]!=b[:52]))
            np.testing.assert_array_equal(a[55:],b[55:])
            self.assertEqual(preparar_leyenda(ruta,'Baja el logo',None,None),out)

    def test_textura_solo_si_la_muestra_es_del_video_y_tiene_detalle(self):
        im=Image.new('RGB',(120,155),'white')
        self.assertIsNone(muestra_de_textura(im,(30,25,52,40)))
        rng=np.random.default_rng(42)
        im=Image.fromarray(rng.integers(20,170,(155,120,3),dtype=np.uint8))
        self.assertGreater(muestra_de_textura(im,(30,25,52,40)).std(),.005)
