"""Presentación de citas web en el chat compartido por la web y la voz."""
import re
import unicodedata

SECCION = "\n\nFuentes encontradas:\n"
_OBJETO = r"(?:urls?|enlaces?|links?|fuentes?|referencias?|direcci[oó]n web)"
_PETICION = r"(?:(?:me )?(?:das|pasas|muestras|envias|compartes)|(?:me )?(?:puedes|podrias) (?:dar|pasar|mostrar|compartir|enviar)(?:me)?|(?:dame|pasame|muestrame|enviame|comparte(?:me)?|incluye|cita|quiero|necesito)(?: ver)?|cual(?:es)? (?:es|son|fue|fueron))"
_ARTICULO = r"(?:(?:el|la|los|las|un|una|unos|unas|tu|tus|esa|esas|ese|esos) )?"
_URL = r"(?:https?://|www\.)[^\s<>()\[\]]+(?:\([^\s()]*\)[^\s<>()\[\]]*)*"
_MARCA = re.compile(r"\[([^\]\n]*)\]\s*\(\s*(" + _URL + r")\s*\)", re.I)


def pedido_fuentes(pregunta):
    """Devuelve si pide enlaces y si se refiere solo a la respuesta anterior."""
    texto = unicodedata.normalize("NFKD", pregunta.casefold())
    texto = " ".join(re.findall(r"\w+", "".join(c for c in texto if not unicodedata.combining(c))))
    texto = re.sub(r"^(?:aura |por favor |y )+|(?: por favor| gracias)$", "", texto)
    patron = _PETICION + " " + _ARTICULO + _OBJETO
    origen = r"(?:de )?donde (?:lo |la )?(?:sacaste|encontraste|consultaste)(?: eso| esa informacion)?"
    previa = bool(re.fullmatch(r"(?:(?:" + patron + "|" + _ARTICULO + _OBJETO +
                               r")(?: (?:original(?:es)?|complet[oa]s?|de (?:eso|esa respuesta|esa informacion)))?|" + origen + ")", texto))
    if re.search(r"\b(?:sin|no (?:quiero|incluyas|muestres|leas|me des)) " + _ARTICULO + _OBJETO, texto):
        return False, False
    explicita = re.search(r"\b(?:" + patron + "|con " + _ARTICULO + _OBJETO + "|" + origen + r")\b", texto)
    return bool(previa or explicita), previa


def sin_citas(texto):
    original = texto
    texto = re.sub(r"(?im)^\s*(?:fuentes?|enlaces?) disponibles?[.!]?\s*$", "", texto)
    texto = texto.partition(SECCION)[0]
    # Las referencias parentéticas (incluidas las de la captura) se retiran enteras.
    texto = re.sub(r"\(\s*\[[^\]\n]*\]\s*\(\s*" + _URL + r"\s*\)\s*\)", "", texto, flags=re.I)
    # Conserva el significado de un enlace integrado en una oración.
    def etiqueta(m):
        nombre = m[1].strip()
        host = re.split(r"[/?:#]", re.sub(r"^(?:https?://)?(?:www\.)?", "", m[2], flags=re.I), 1)[0]
        if nombre.casefold() in host.casefold().split("."):
            return ""
        if re.fullmatch(r"(?:fuente|enlace|link|\d+|[^\s]+\.[a-z]{2,}(?:/\S*)?)", nombre, re.I):
            return ""
        return nombre
    texto = _MARCA.sub(etiqueta, texto)
    texto = re.sub(r"\(\s*(?:fuente\s*:\s*)?" + _URL + r"\s*\)", "", texto, flags=re.I)
    texto = re.sub(_URL, lambda m: re.search(r"[.,;:!?]*$", m[0])[0], texto, flags=re.I)
    texto = re.sub(r"[ \t]*[]cite[][^\n]*[]", "", texto)
    if texto != original:
        texto = re.sub(r"[ \t]+([.,;:!?])", r"\1", texto)
        texto = re.sub(r"(?<=\S)[ \t]{2,}", " ", texto)
        texto = re.sub(r"(?m)^\s*(?:[-*]\s*)?(?:fuentes?(?: encontradas)?|referencias?|enlaces?)\s*:\s*$", "", texto)
    return texto.strip() or "No pude obtener un resumen de esa consulta."
