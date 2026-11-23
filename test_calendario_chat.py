import unittest
from datetime import datetime

from calendario_chat import contexto_fecha, hora_del_chat, respuesta_calendario


class CalendarioChatTest(unittest.TestCase):
    def test_pregunta_y_rectificacion_la_noche_del_domingo(self):
        instante = datetime.fromisoformat("2026-11-23T00:06:00+00:00")
        nueva_york = hora_del_chat("America/New_York", instante)
        self.assertEqual(respuesta_calendario("hola aura quisiera saber manana que dia estamos", nueva_york),
                         "Mañana será lunes 23 de noviembre de 2026.")
        self.assertEqual(respuesta_calendario("segura que hoy es lunes", nueva_york),
                         "No, hoy es domingo 22 de noviembre de 2026.")
        self.assertEqual(respuesta_calendario("estas segura de que hoy es domingo", nueva_york),
                         "Sí, hoy es domingo 22 de noviembre de 2026.")
        madrid = hora_del_chat("Europe/Madrid", instante)
        self.assertEqual(respuesta_calendario("es lunes hoy", madrid), "Sí, hoy es lunes 23 de noviembre de 2026.")
        self.assertIn("hoy es domingo 22 de noviembre de 2026", contexto_fecha(nueva_york))

    def test_medianoche_fin_de_ano_bisiesto_y_cambio_de_hora(self):
        for instante, pregunta, esperado in [
            ("2026-11-23T05:00:00+00:00", "que fecha es hoy", "Hoy es lunes 23 de noviembre de 2026."),
            ("2027-01-01T02:00:00+00:00", "que dia sera manana", "Mañana será viernes 1 de enero de 2027."),
            ("2028-03-01T05:00:00+00:00", "ayer que dia fue", "Ayer fue martes 29 de febrero de 2028."),
            ("2026-03-08T05:30:00+00:00", "que dia sera pasado manana", "Pasado mañana será martes 10 de marzo de 2026."),
            ("2026-11-01T04:30:00+00:00", "que dia sera manana", "Mañana será lunes 2 de noviembre de 2026."),
        ]:
            with self.subTest(instante=instante):
                ahora = hora_del_chat("America/New_York", datetime.fromisoformat(instante))
                self.assertEqual(respuesta_calendario(pregunta, ahora), esperado)

    def test_consultas_con_otro_proposito_y_clientes_sin_zona(self):
        for zona in [None, "Una/ZonaInvalida", "../utc"]:
            ahora = hora_del_chat(zona, datetime.fromisoformat("2026-11-23T00:06:00+00:00"))
            self.assertEqual(respuesta_calendario("en que fecha estamos", ahora), "Hoy es lunes 23 de noviembre de 2026 (UTC).")
        for consulta in ["que clima hara manana", "que dia es navidad", "que dia es hoy en tokio",
                         "si hoy es lunes que dia sera en diez dias", "que fecha tiene la tarea",
                         "que dia es manana y que tiempo hara", "que dia es la reunion de manana"]:
            with self.subTest(consulta=consulta):
                self.assertIsNone(respuesta_calendario(consulta, ahora))


if __name__ == "__main__":
    unittest.main()
