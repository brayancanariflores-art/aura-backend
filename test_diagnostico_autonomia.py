import sqlite3
import tempfile
import unittest
from pathlib import Path

from diagnostico_autonomia import diagnosticar


class DiagnosticoTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        carpeta = Path(temp.name)
        self.historial, self.memoria, self.agente = [carpeta / nombre for nombre in
                                                   ("historial.db", "memoria.db", "agente #1.db")]
        for ruta in (self.historial, self.memoria):
            sqlite3.connect(ruta).close()

    def consultar(self):
        return diagnosticar(self.historial, self.memoria, self.agente, True, True)

    def test_agente_ausente(self):
        estado = self.consultar()
        self.assertFalse(self.agente.exists())
        self.assertFalse(estado["preparacion_basica"])
        self.assertIsNone(estado["objetivos_activos"])

    def test_esquema_incompleto(self):
        sqlite3.connect(self.agente).close()
        estado = self.consultar()
        self.assertTrue(estado["bases"]["agente"]["integridad"])
        self.assertFalse(estado["preparacion_basica"])
        self.assertIsNone(estado["ciclos_pendientes_revision"])

    def test_contadores_reales_y_lectura_sin_cambios(self):
        conn = sqlite3.connect(self.agente)
        conn.executescript("""
            CREATE TABLE objetivos_autonomos (activo INTEGER);
            CREATE TABLE ciclos_autonomos (estado TEXT);
            INSERT INTO objetivos_autonomos VALUES (1), (0), (1);
            INSERT INTO ciclos_autonomos VALUES ('pendiente_revision_humana'), ('otro');
        """)
        conn.close()
        anterior = self.agente.read_bytes()
        estado = self.consultar()
        self.assertTrue(estado["preparacion_basica"])
        self.assertEqual(estado["objetivos_activos"], 2)
        self.assertEqual(estado["ciclos_pendientes_revision"], 1)
        self.assertEqual(anterior, self.agente.read_bytes())
