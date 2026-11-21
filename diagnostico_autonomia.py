import sqlite3
from contextlib import closing
from pathlib import Path


def diagnosticar(historial, memoria, agente, modelo_configurado, autenticacion_configurada):
    """Comprueba las bases existentes y los contadores sin crear bases ausentes."""
    bases = {}
    objetivos = pendientes = None
    for nombre, ruta in (("historial", historial), ("memoria", memoria), ("agente", agente)):
        uri = Path(ruta).resolve().as_uri() + "?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True, timeout=3)) as conn:
                integro = conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                bases[nombre] = {"accesible": True, "integridad": integro}
                if nombre == "agente" and integro:
                    objetivos, pendientes = conn.execute("""
                        SELECT
                          (SELECT COUNT(*) FROM objetivos_autonomos WHERE activo = 1),
                          (SELECT COUNT(*) FROM ciclos_autonomos WHERE estado = 'pendiente_revision_humana')
                    """).fetchone()
        except (sqlite3.Error, OSError) as exc:
            if nombre not in bases:
                bases[nombre] = {"accesible": False, "integridad": False}
            bases[nombre]["tipo_error"] = type(exc).__name__
    preparado = (modelo_configurado and autenticacion_configurada and objetivos is not None
                 and all(base["integridad"] for base in bases.values()))
    return {
        'preparacion_basica': preparado,
        'modelo_configurado': modelo_configurado,
        'autenticacion_configurada': autenticacion_configurada,
        'bases': bases,
        'objetivos_activos': objetivos,
        'ciclos_pendientes_revision': pendientes,
        'ejecucion_continua': 'requiere_disparador_externo',
        'ejecucion_externa_automatica': False,
        'advertencia': 'Diagnostico basico; no certifica seguridad ni disponibilidad en produccion.'
    }
