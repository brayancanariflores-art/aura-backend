"""Fechas de calendario para el chat, independientes del idioma del servidor."""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre")
REFERENCIAS = {
    "anteayer": (-2, "anteayer fue"), "ayer": (-1, "ayer fue"),
    "hoy": (0, "hoy es"), "manana": (1, "mañana será"),
    "pasado manana": (2, "pasado mañana será"),
}
MOMENTO = r"pasado manana|manana|anteayer|ayer|hoy"
DIA_SEMANA = r"lunes|martes|miercoles|jueves|viernes|sabado|domingo"
PREGUNTA = re.compile(
    r"(?:que (?:dia(?: de la semana)?|fecha)(?: (?:es|sera|fue|era|cae|estamos|estaremos))?"
    r"|(?:en|a) que (?:dia|fecha) (?:estamos|estaremos)"
    r"|cual es la fecha|(?:la )?fecha)"
)
SALUDO = re.compile(
    r"^(?:(?:hola|oye|aura|y|buenos dias|buenas tardes|buenas noches|por favor|"
    r"quisiera saber|quiero saber|me gustaria saber|me puedes decir|puedes decirme|dime|"
    r"(?:estas )?segur[oa](?: de)? que|es (?:cierto|verdad) que) )+"
)


def hora_del_chat(zona, instante=None):
    instante = instante or datetime.now(timezone.utc)
    try:
        tz = ZoneInfo(zona) if zona else timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        tz = timezone.utc
    return instante.astimezone(tz)


def fecha_larga(fecha):
    return f"{DIAS[fecha.weekday()]} {fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}"


def contexto_fecha(ahora):
    hoy = ahora.date()
    return (f"Referencia de calendario ({ahora.tzinfo}): hoy es {fecha_larga(hoy)}; "
            f"mañana será {fecha_larga(hoy + timedelta(days=1))}. ")


def respuesta_calendario(normal, ahora):
    """Recibe la pregunta normalizada y solo resuelve consultas completas de calendario."""
    consulta = SALUDO.sub("", normal).removesuffix(" por favor")
    principio = re.match(rf"^({MOMENTO}) ", consulta)
    final = re.search(rf" ({MOMENTO})$", consulta)
    if principio and final:
        return None
    referencia = "hoy"
    if principio:
        referencia = principio[1]
        consulta = consulta[principio.end():]
    elif final:
        referencia = final[1]
        consulta = consulta[:final.start()].removesuffix(" de")

    confirma = re.fullmatch(rf"(?:es|sera|fue|era) ({DIA_SEMANA})", consulta)
    if not confirma and not PREGUNTA.fullmatch(consulta):
        return None

    dias, inicio = REFERENCIAS[referencia]
    fecha = ahora.date() + timedelta(days=dias)
    prefijo = ""
    if confirma:
        dia = DIAS[fecha.weekday()].replace("é", "e").replace("á", "a")
        prefijo = "Sí, " if confirma[1] == dia else "No, "
    zona = " (UTC)" if str(ahora.tzinfo) == "UTC" else ""
    return f"{prefijo}{inicio if prefijo else inicio.capitalize()} {fecha_larga(fecha)}{zona}."
