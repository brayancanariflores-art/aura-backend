import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from marca_video import validar_cajas
from reels_chat import preparar_reel


class MarcaVideoTest(unittest.TestCase):
    def test_coordenadas_del_video_y_tramos_visibles(self):
        # Video horizontal en celdas de 216 x 312, con bandas arriba y abajo.
        cajas = [[1, 54, 129, 108, 183], [3, 60, 130, 120, 180]]
        grupos = validar_cajas(cajas, 2, 4, 400, 200, 2)
        self.assertEqual(len(grupos), 2)
        self.assertEqual(grupos[0][1], 1)
        for real, pedido in zip(grupos[0][0][0], [0.5, 100, 50, 100, 100]):
            self.assertAlmostEqual(real, pedido)

    def test_marca_incierta_o_fuera_del_video(self):
        for cajas in ([], [[0, 10, 0, 30, 30]], [[3, 0, 100, 60, 160]]):
            with self.subTest(cajas=cajas), self.assertRaises(ValueError):
                validar_cajas(cajas, 2, 2, 400, 200, 1)

    def test_edicion_usa_los_adjuntos_sin_generar_escenas(self):
        with TemporaryDirectory() as carpeta, patch("reels_chat.editar_marca") as editar:
            base = Path(carpeta)
            (base / "marca.png").touch()
            cliente = Mock()
            self.assertIs(preparar_reel(base,"Cambia el logo del vaso",cliente,"modelo"), editar.return_value)
            editar.assert_called_once_with(base,"Cambia el logo del vaso",cliente,"modelo")
            self.assertFalse(cliente.mock_calls)

    def test_guia_porcentual_seguida_hasta_su_ultimo_punto(self):
        from marca_video import trayectoria_guiada
        guia = "Reemplaza el logo. Guía de movimiento: [[1,10,20,30,50],[2,20,20,45,60]]"
        self.assertEqual(trayectoria_guiada(guia, 200, 100, 3),
                         [([[1, 20, 20, 40, 30], [2, 40, 20, 50, 40]], 2)])
        self.assertIsNone(trayectoria_guiada("Reemplaza la marca del vaso", 200, 100, 3))

    def test_guia_invalida_no_produce_una_edicion(self):
        from marca_video import trayectoria_guiada
        for guia in ("", "{}", "[[1,0,0,40,35],[1,0,0,40,35]]",
                     "[[0,0,0,10,10],[2,-5,10,8,20]]"):
            with self.subTest(guia=guia), self.assertRaises(ValueError):
                trayectoria_guiada("Guía de movimiento: " + guia, 200, 100, 3)
