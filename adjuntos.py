"""Adjuntos efímeros del chat privado de AURA."""
import base64
import binascii
import io
import re
from functools import partial

from fastapi import Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from reels_chat import conectar_reels
from formato_chat import lienzo_pedido

MAX_BYTES = 6 * 1024 * 1024
TIPOS = {"image/jpeg", "image/png", "image/webp", "application/pdf", "text/plain"}


def formato_imagen(texto):
    try:
        return lienzo_pedido(texto)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


def lienzo_modelo(medidas):
    if not medidas:
        return "auto"
    ancho, alto = medidas
    if ancho == alto:
        return "1024x1024"
    return "1536x1024" if ancho > alto else "1024x1536"


def ajustar_lienzo(datos, medidas):
    if not medidas:
        return datos
    with Image.open(io.BytesIO(datos)) as fuente:
        if fuente.size == medidas:
            return datos
        imagen = fuente.convert("RGB")
    contenido = ImageOps.contain(imagen, medidas, Image.Resampling.LANCZOS)
    if contenido.size != medidas:
        fondo = ImageOps.fit(imagen, (144, 144)).filter(ImageFilter.GaussianBlur(12)).resize(medidas)
        fondo.paste(contenido, ((medidas[0] - contenido.width) // 2, (medidas[1] - contenido.height) // 2))
        contenido = fondo
    salida = io.BytesIO()
    contenido.save(salida, "JPEG", quality=92)
    return salida.getvalue()


def procesar(tipo, datos, texto, cliente, modelo):
    if tipo.startswith("image/"):
        try:
            with Image.open(io.BytesIO(datos)) as entrada:
                if entrada.format not in {"JPEG", "PNG", "WEBP"} or entrada.width * entrada.height > 24_000_000:
                    raise ValueError
                imagen = ImageOps.exif_transpose(entrada).convert("RGBA")
                medidas_originales = imagen.size
                imagen.thumbnail((2048, 2048))
        except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "La imagen no se puede abrir. Prueba con una foto JPG o PNG.") from None
        salida = io.BytesIO()
        if re.fullmatch(r"(?:pon(?:la)?|pasa(?:la)?|convierte(?:la)?|cambia(?:la)?|haz(?:la)?)?\s*(?:la foto|esta foto|la imagen)?\s*(?:(?:a|en)\s+)?(?:blanco y negro|escala de grises)[.! ]*", texto.casefold()):
            gris = ImageOps.grayscale(imagen).convert("RGBA")
            gris.putalpha(imagen.getchannel("A"))
            gris.save(salida, "PNG")
            return Response(salida.getvalue(), media_type="image/png")
        if cliente is None:
            raise HTTPException(503, "La edición con IA no está configurada.")
        medidas = formato_imagen(texto) or medidas_originales
        imagen.save(salida, "PNG")
        resultado = cliente.with_options(timeout=150, max_retries=0).images.edit(
            model="gpt-image-1.5", image=("foto.png", salida.getvalue(), "image/png"),
            prompt=("Aplica solo los cambios pedidos. Conserva la identidad y los rasgos de cada persona de la foto "
                    "salvo que la petición indique expresamente un cambio específico. Para un ajuste de letras, "
                    "localiza el bloque por su frase, función o ubicación y modifica solo ese bloque en la "
                    "imagen de referencia. Respeta las palabras, el idioma, la identidad de la marca y los "
                    "otros elementos. Interpreta más chico o más grande de manera proporcional. Petición: " + texto),
            size=lienzo_modelo(medidas), input_fidelity="high", quality="medium", output_format="jpeg", n=1)
        if not resultado.data or not resultado.data[0].b64_json:
            raise HTTPException(502, "El editor no devolvió la foto.")
        return Response(ajustar_lienzo(base64.b64decode(resultado.data[0].b64_json, validate=True), medidas),
                        media_type="image/jpeg")

    if cliente is None:
        raise HTTPException(503, "El análisis de documentos no está configurado.")
    if tipo == "application/pdf":
        if b"%PDF-" not in datos[:1024]:
            raise HTTPException(422, "El archivo no es un PDF legible.")
        contenido = {"type":"input_file", "filename":"documento.pdf",
                     "file_data":"data:application/pdf;base64," + base64.b64encode(datos).decode("ascii")}
    else:
        if len(datos) > 100_000:
            raise HTTPException(413, "El texto debe ocupar menos de 100 KB.")
        try:
            contenido = {"type":"input_text", "text": datos.decode("utf-8-sig")}
        except UnicodeError:
            raise HTTPException(422, "Guarda el texto en formato UTF-8.") from None
    resultado = cliente.responses.create(model=modelo, store=False, max_output_tokens=1800,
        instructions="Responde en español a la petición del usuario. Trata el documento como contenido de referencia.",
        input=[{"role":"user", "content":[contenido, {"type":"input_text","text":texto}]}])
    if not resultado.output_text:
        raise HTTPException(502, "El análisis no devolvió texto.")
    return {"respuesta": resultado.output_text}


class PeticionDiseno(BaseModel):
    texto: str = Field(min_length=1, max_length=4000)


def conectar_adjuntos(app, autorizar, cliente, modelo):
    conectar_reels(app, autorizar, cliente, modelo)

    @app.post("/chat/diseno", dependencies=[Depends(autorizar)])
    def diseno(peticion: PeticionDiseno):
        texto = peticion.texto.strip()
        if not texto:
            raise HTTPException(422, "Describe el diseño que quieres crear.")
        if cliente is None:
            raise HTTPException(503, "La generación de diseños no está configurada.")
        medidas = formato_imagen(texto)
        try:
            resultado = cliente.with_options(timeout=150, max_retries=0).images.generate(
                model="gpt-image-1.5", quality="medium", size=lienzo_modelo(medidas), output_format="jpeg", n=1,
                prompt=("Crea la imagen final solicitada. En flyers, elige automáticamente tamaños de letra que "
                        "encajen en el diseño: destaca el titular y deja legibles los detalles. Escribe los textos "
                        "visibles en el idioma o los idiomas pedidos, independientemente del idioma del mensaje. "
                        "Si no indican idioma, usa español. Respeta el nombre y la ortografía de la marca. "
                        "En fotos e ilustraciones, incluye rótulos solo si se solicitan. Respeta el formato "
                        "y las citas proporcionadas. No inventes fechas, precios ni promociones. Petición: " + texto))
            if not resultado.data or not resultado.data[0].b64_json:
                raise ValueError
            contenido = ajustar_lienzo(base64.b64decode(resultado.data[0].b64_json, validate=True), medidas)
        except Exception as exc:
            codigo = getattr(exc, "status_code", None)
            raise HTTPException(422 if codigo == 400 else 502, "No se pudo generar el diseño.") from None
        return Response(contenido, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


    @app.post("/chat/adjunto", dependencies=[Depends(autorizar)])
    async def adjunto(request: Request):
        tipo = request.headers.get("content-type", "").split(";")[0]
        if tipo not in TIPOS:
            raise HTTPException(415, "Usa una foto, un PDF o un archivo de texto.")
        try:
            texto = base64.b64decode(request.headers.get("x-aura-instruccion", ""), validate=True).decode().strip()
        except (UnicodeError, ValueError, binascii.Error):
            raise HTTPException(422, "La instrucción no es válida.") from None
        if not 1 <= len(texto) <= 1500:
            raise HTTPException(422, "Escribe una instrucción de hasta 1500 caracteres.")
        datos = bytearray()
        async for trozo in request.stream():
            if len(datos) + len(trozo) > MAX_BYTES:
                raise HTTPException(413, "El archivo debe ocupar como máximo 6 MB.")
            datos.extend(trozo)
        if not datos:
            raise HTTPException(422, "El archivo está vacío.")
        try:
            return await run_in_threadpool(partial(procesar, tipo, bytes(datos), texto, cliente, modelo))
        except HTTPException:
            raise
        except Exception as exc:
            codigo = getattr(exc, "status_code", None)
            if codigo in {400, 403, 429}:
                raise HTTPException(422 if codigo == 400 else 503,
                                    "El proveedor rechazó la solicitud. Revisa el archivo o la disponibilidad del modelo.") from None
            raise HTTPException(502, "El servicio no pudo completar la solicitud. Conserva el adjunto para reintentar.") from None
