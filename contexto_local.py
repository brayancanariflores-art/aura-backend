import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field


class ContextoDispositivo(BaseModel):
    zona: str = Field(default="", max_length=100)
    region: str = Field(default="", max_length=100)
    pais: str = Field(default="", max_length=2)


def contexto_del_turno(dispositivo, ahora=None):
    instante = ahora or datetime.now(timezone.utc)
    zona = None
    if dispositivo and dispositivo.zona:
        try:
            zona = ZoneInfo(dispositivo.zona)
        except (ValueError, ZoneInfoNotFoundError):
            pass
    local = instante.astimezone(zona or timezone.utc)
    datos = {"fecha": local.date().isoformat(), "hora": local.strftime("%H:%M"),
             "zona": zona.key if zona else "UTC", "zona_del_dispositivo": zona is not None}
    if dispositivo and re.fullmatch(r"[A-Z]{2}", dispositivo.pais):
        datos["pais"] = dispositivo.pais
        region = dispositivo.region.strip()
        if region and all(c.isalpha() or c in " -.'’" for c in region):
            datos["region"] = region
    return datos


def respuesta_local(pregunta_normalizada, datos):
    texto = re.sub(r"^(?:hola )?aura ", "", pregunta_normalizada)
    texto = re.sub(r"^(?:me puedes decir|puedes decirme|dime|me dices) ", "", texto)
    texto = texto.removesuffix(" por favor")
    zona = datos["zona"]
    if re.fullmatch(r"(?:que|cual es la|dime la|la) fecha y (?:que |la )?hora(?: (?:es|son|tenemos|actual|actuales|hoy|aqui))*", texto):
        return respuesta_local("que fecha es", datos) + " " + respuesta_local("que hora es", datos)
    if re.fullmatch(r"(?:que hora es|que horas son|la hora)(?: ahora| aqui| local| en mi ubicacion)*", texto):
        return (f"Son las {datos['hora']} ({zona})." if datos["zona_del_dispositivo"] else
                f"Son las {datos['hora']} UTC. ¿En qué zona horaria estás?")
    if re.fullmatch(r"(?:que fecha es|cual es la fecha|que dia es|a que fecha estamos|la fecha)(?: de)?(?: hoy)?", texto):
        fecha = datetime.fromisoformat(datos["fecha"])
        dias = "lunes martes miércoles jueves viernes sábado domingo".split()
        meses = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
        referencia = "" if datos["zona_del_dispositivo"] else " en UTC"
        return f"Hoy{referencia} es {dias[fecha.weekday()]} {fecha.day} de {meses[fecha.month-1]} de {fecha.year}."
    if re.fullmatch(r"(?:donde (?:estoy|me encuentro)|(?:en que (?:estado|region|pais)) (?:estoy|me encuentro))(?: ahora| mismo)*", texto):
        ubicacion = ", ".join(datos[k] for k in ("region", "pais") if k in datos)
        return (f"Tu ubicación aproximada es {ubicacion}." if ubicacion else
                "Aún no tengo tu región. Puedes decírmela o permitir la ubicación aproximada en el iPhone.")
    return None


def es_consulta_electoral(texto):
    return bool(re.search(r"\b(?:elecciones|eleccion presidencial|resultados electorales|comicios)\b", texto) or (
        re.search(r"\b(?:democratas?|republicanos?|partido politico|candidat[oa]s?)\b", texto)
        and re.search(r"\b(?:gana\w*|gan[eo]|triunf\w*|victoria|vot\w*|escanos|mayoria)\b", texto)
    ))


def referencia_local(datos):
    return ("Datos de referencia de este turno: " + json.dumps(datos, ensure_ascii=False) +
            ". La zona horaria por sí sola no indica dónde está el usuario. "
            "Usa la región compartida para referencias locales; da prioridad al lugar y período "
            "que el usuario indique. Si falta la región para responder, pregunta.")
