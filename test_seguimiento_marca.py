import unittest

import numpy as np

from seguimiento_marca import anclas, canales, seguir, trayectoria_adaptada


class SeguimientoMarcaTest(unittest.TestCase):
    def test_sigue_tinta_de_dos_colores_y_se_detiene_al_desaparecer(self):
        for tono, rgb in [(80, [44, 220, 32]), (160, [35, 80, 220])]:
            with self.subTest(tono=tono):
                frames = np.full((25, 120, 136, 3), 115, dtype=np.uint8)
                for i, frame in enumerate(frames):
                    if i >= 5:
                        for dx in range(0, 28, 6):
                            frame[44:64, 25+i+dx:28+i+dx] = rgb
                candidatos = anclas(canales(frames[-1]))
                candidatos = [p for p in candidatos if abs(p[1]-tono)<=16]
                self.assertTrue(candidatos)
                _, color, inicial = candidatos[0]
                pista = seguir(frames, 24, color, inicial)
                self.assertEqual(sorted(pista), list(range(5,25)))
                self.assertLess(abs(pista[6][0]-31), 3)
                self.assertLess(abs(pista[21][0]-46), 3)
                grupo, fin = trayectoria_adaptada(pista, "ambos", 15, (272,240),
                                                  (120,136), 25/15)[0]
                self.assertAlmostEqual(grupo[0][0],5/15)
                self.assertAlmostEqual(fin,25/15)

    def test_fondo_neutro_sin_marca(self):
        im = np.full((120,180,3), 100, dtype=np.uint8)
        self.assertFalse(anclas(canales(im)))
