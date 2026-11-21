import unittest
from datetime import datetime, timezone

from contexto_local import ContextoDispositivo, contexto_del_turno, es_consulta_electoral, respuesta_local


class ContextoLocalTest(unittest.TestCase):
    def test_cambio_de_fecha_y_horario_de_verano(self):
        dispositivo = ContextoDispositivo(zona="America/New_York")
        for mes, hora in [(1, "21:30"), (7, "22:30")]:
            instante = datetime(2026, mes, 2, 2, 30, tzinfo=timezone.utc)
            datos = contexto_del_turno(dispositivo, instante)
            self.assertEqual(datos["fecha"], f"2026-{mes:02d}-01")
            self.assertEqual(datos["hora"], hora)
            self.assertNotIn("region", datos)
            self.assertIn(hora, respuesta_local("que fecha y hora es hoy", datos))

    def test_region_es_opcional_y_no_se_deduce_de_la_zona(self):
        for zona in ["America/New_York", "Zona/Falsa", "../localtime"]:
            datos = contexto_del_turno(ContextoDispositivo(zona=zona))
            self.assertIn("Aún no tengo tu región", respuesta_local("donde estoy", datos))
        self.assertFalse(contexto_del_turno(None)["zona_del_dispositivo"])
        datos = contexto_del_turno(ContextoDispositivo(region="Nueva Jersey", pais="US"))
        self.assertIn("Nueva Jersey", respuesta_local("en que estado estoy", datos))
        self.assertIsNone(respuesta_local("que hora es en paris", datos))

    def test_elecciones_sin_decir_noticias(self):
        self.assertTrue(es_consulta_electoral("que piensas de que hayan ganado los democratas"))
        self.assertTrue(es_consulta_electoral("quien gano las elecciones en nueva jersey"))
        self.assertFalse(es_consulta_electoral("quien gano el partido de futbol"))
