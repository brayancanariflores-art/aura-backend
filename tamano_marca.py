"""Tamaño de la nueva impresión, relativo al tamaño inicial del arte."""
import re
import unicodedata

UNIDADES = "cero uno dos tres cuatro cinco seis siete ocho nueve".split()
NUMEROS = {nombre: i for i, nombre in enumerate(UNIDADES)}
NUMEROS.update(dict(zip("diez once doce trece catorce quince dieciseis diecisiete dieciocho diecinueve".split(), range(10, 20))))
for i, decena in enumerate("veinte treinta cuarenta cincuenta sesenta setenta ochenta noventa".split(), 2):
    NUMEROS[decena] = i * 10
    for j, unidad in enumerate(UNIDADES[1:], 1):
        nombre = "veinti" + unidad if i == 2 else decena + " y " + unidad
        NUMEROS[nombre] = i * 10 + j
NUMEROS["cien"] = 100
NUMERO = r"\d+(?:[.,]\d+)?|" + "|".join(sorted(NUMEROS, key=len, reverse=True))
PORCENTAJE = re.compile(rf"\b({NUMERO})\s*(?:%|por ciento\b)")
OBJETO = r"(?:(?:el|la|su)\s+)?(?:tamano(?:\s+del?)?\s*)?(?:logo(?:tipo)?|marca(?: de agua)?|imagen|sello|impresion|arte)?"
MAYOR = r"agrand\w*|ampli\w*|aument\w*"
MENOR = r"achic\w*|reduc\w*|disminu\w*"
VERBO = re.compile(rf"\b({MAYOR}|{MENOR})\s*{OBJETO}\s*(?:(?:a|en|al|un)\s+)*$")
COMPARACION = re.compile(r"^\s*mas\s+(grande|pequen[oa]|chic[oa])\b")


def normalizar(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if not unicodedata.combining(c))


def instruccion_tamano(texto):
    texto = normalizar(texto)
    for pct in reversed(list(PORCENTAJE.finditer(texto))):
        antes, despues = texto[:pct.start()], texto[pct.end():]
        # En una indicación como «treinta o treinta y cinco», elegir el último.
        antes = re.sub(rf"\b(?:{NUMERO})\s+(?:o|a)\s+$", "", antes)
        orden = VERBO.search(antes)
        comparacion = COMPARACION.match(despues)
        if comparacion is None and orden is None:
            continue
        if not comparacion and re.match(r"\s*(?:de\s+)?(?:opacidad|transparencia|brillo|volumen|audio)\b", despues):
            continue
        menor = comparacion[1] != "grande" if comparacion else re.fullmatch(MENOR, orden[1]) is not None
        n = pct[1]
        valor = NUMEROS[n] if n in NUMEROS else float(n.replace(",", "."))
        factor = 1 + (-valor if menor else valor)/100
        if not 0 < factor:
            continue
        return factor
    if re.search(r"(?:tamano|escala) (?:original|inicial|normal)\b", texto):
        return 1.
    return None


def escala_marca(texto):
    for pedido in reversed(texto.lower().split("ajuste solicitado:")):
        factor = instruccion_tamano(pedido)
        if factor is not None:
            return factor
    return 1.


def solo_tamano(texto):
    # Reutilizar la trayectoria únicamente ante una revisión simple de escala.
    pedido = normalizar(texto).strip(" ¡!¿?.,")
    numero = rf"(?:{NUMERO})(?:\s+o\s+(?:{NUMERO}))?"
    porcentaje = rf"{numero}\s*(?:%|por ciento)"
    prefijo = r"(?:(?:por favor|ahora|solo|solamente)\s+)*"
    verbo = rf"(?:{MAYOR}|{MENOR})\s*{OBJETO}\s*(?:(?:a|en|al|un)\s+)*"
    comparacion = rf"(?:haz(?:lo|la)?\s+)?{OBJETO}\s*(?:un\s+)?{porcentaje}\s+mas (?:grande|pequen[oa]|chic[oa])"
    return bool(re.fullmatch(rf"{prefijo}(?:{verbo}{porcentaje}|{comparacion})", pedido))
