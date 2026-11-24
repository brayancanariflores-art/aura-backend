import unittest

from fuentes_chat import SECCION, pedido_fuentes, pide_busqueda_directa, sin_citas

URL = "https://liquipedia.net/dota2/Game_Versions?utm_source=openai"


class FuentesChatTest(unittest.TestCase):
    def test_un_relato_personal_sigue_en_el_chat(self):
        for frase in ("Hoy tuve un turno pesado", "Ayer vi las noticias y me quedé triste",
                      "Hoy me siento nervioso. ¿Qué le digo a mi primo?",
                      "Esta semana terminé el proyecto", "Mi internet anda lento"):
            with self.subTest(frase=frase):
                self.assertFalse(pide_busqueda_directa(frase))
                self.assertEqual(pedido_fuentes(frase), (False, False))

    def test_pedidos_expresos_de_actualidad(self):
        for frase in ("Busca las noticias de hoy", "AURA, busca en internet el horario",
                      "¿Me puedes buscar las últimas noticias?", "Noticias de ayer"):
            with self.subTest(frase=frase):
                self.assertTrue(pide_busqueda_directa(frase))

    def test_omite_la_cita_y_la_lista_sin_perder_texto(self):
        for cita in (f"([liquipedia.net]({URL}))", f"[liquipedia.net]({URL})",
                     f"[Liquipedia]({URL})", f"({URL})", URL):
            with self.subTest(cita=cita):
                texto = f"Primera frase. {cita} Segunda frase." + SECCION + "- Versiones: " + URL
                self.assertEqual(sin_citas(texto), "Primera frase. Segunda frase.")

    def test_conserva_texto_y_formato(self):
        self.assertEqual(sin_citas("Pasos:\n  1. Abre el juego.\n  2. Juega."), "Pasos:\n  1. Abre el juego.\n  2. Juega.")
        self.assertEqual(sin_citas(f"Lee las [notas del parche]({URL})."), "Lee las notas del parche.")
        self.assertEqual(sin_citas("Respuesta.\nFuente disponible."), "Respuesta.")

    def test_intencion(self):
        for orden in ("Pásame el enlace", "¿De dónde lo sacaste?", "¿Me puedes pasar la URL?", "Fuentes, por favor"):
            self.assertEqual(pedido_fuentes(orden), (True, True), orden)
        self.assertEqual(pedido_fuentes("Busca las novedades sin enlaces"), (False, False))
        self.assertEqual(pedido_fuentes("¿Qué son las fuentes renovables?"), (False, False))
        self.assertEqual(pedido_fuentes("Dame el enlace de la página oficial de Dota"), (True, False))
