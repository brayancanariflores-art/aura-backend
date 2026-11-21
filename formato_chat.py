"""Interpreta el lienzo del pedido sin confundirlo con el tamaño de letra."""
import re


def lienzo_pedido(texto, *, video=False):
    pedido = texto.partition("\nContexto de la conversación:")[0].casefold()
    pixeles = re.search(r"\b(\d{2,5})\s*(?:[x×]|por)\s*(\d{2,5})\b", pedido)
    if pixeles:
        ancho, alto = map(int, pixeles.groups())
        limite = 1920 if video else 4096
        if min(ancho, alto) < 32 or max(ancho, alto) > limite:
            raise ValueError(f"Usa un lienzo con lados de entre 32 y {limite} píxeles.")
        if video and (ancho % 2 or alto % 2):
            raise ValueError("Usa dimensiones pares para el video.")
        return ancho, alto
    proporcion = re.search(r"\b(?:formato|proporci[oó]n|aspecto|lienzo)\s+(?:de\s+)?(\d{1,2})\s*[:x×]\s*(\d{1,2})\b", pedido)
    if not proporcion:
        proporcion = re.search(r"\b(9|16|1|4|5|3|2|21)\s*[:x×]\s*(16|9|1|4|5|3|2)\b", pedido)
    if proporcion:
        ancho, alto = map(int, proporcion.groups())
        if min(ancho, alto) < 1 or max(ancho, alto) > min(ancho, alto) * 12:
            raise ValueError("La proporción solicitada está fuera del rango admitido.")
        escala = (1280 if video else 1536) / max(ancho, alto)
        return 2 * round(ancho * escala / 2), 2 * round(alto * escala / 2)
    orientacion = re.search(r"\b(vertical|horizontal|cuadrad[oa])\b", pedido)
    if orientacion:
        corto, largo, lado = (720, 1280, 960) if video else (1024, 1536, 1024)
        return {"vertical": (corto, largo), "horizontal": (largo, corto)}.get(orientacion[0], (lado, lado))
    return None
