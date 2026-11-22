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

from uuid import UUID
from ediciones_chat import ejecutar_edicion, cancelar
from reels_chat import conectar_reels
from formato_chat import lienzo_pedido

MAX_BYTES = 6 * 1024 * 1024
VIDEOS = {"video/mp4", "video/quicktime", "video/webm"}
TIPOS = VIDEOS | {"image/jpeg", "image/png", "image/webp", "application/pdf", "text/plain"}

GUIA_CONSULTA = (
    "Responde en español a la petición. El archivo es material de referencia. Si contiene ejercicios "
    "y el usuario pide ayuda para resolverlos, resuelve cada apartado solicitado. Lee del enunciado y "
    "del diagrama las incógnitas, datos, unidades, posiciones y sentidos. No confundas el valor máximo "
    "de una gráfica con el de su extremo. Haz las operaciones, comprueba los resultados y muestra la "
    "respuesta final por ejercicio, con unidades y convención de signos cuando correspondan. Si falta "
    "un dato, especifica cuál y resuelve lo que sí se pueda. Si solo pide pistas, respeta ese límite. "
    "Escribe fórmulas legibles en texto plano, sin delimitadores LaTeX. Para fotogramas con tiempos, "
    "descríbelos en orden y distingue lo visible de lo que no se puede deducir, incluido el audio."
)


class Solucion(BaseModel):
    ejercicio: str
    calculo: str
    resultado_o_dato_faltante: str
    comprobacion: str


class Soluciones(BaseModel):
    ejercicios: list[Solucion]


def consultar_adjunto(cliente, modelo, contenido, texto):
    entrada = [{"role": "user", "content": [contenido, {"type": "input_text", "text": texto}]}]
    resolver = re.search(r"\b(?:resu[eé]lv\w*|resol\w+|soluci[oó]n\w*|soluciona\w*|ejercicios?|"
                         r"problemas? (?:de |matem)|calc[uú]l[ae]\w*|resultados?|ecuaci[oó]n(?:es)?)\b", texto, re.I)
    peticion = texto.partition("\nConsulta anterior sobre este archivo:")[0]
    solo_lectura = re.search(r"\b(?:pistas?|traduce|transcribe|qu[eé] dice|sin (?:resolver|dar\w* (?:la|el) (?:resultado|respuesta))|"
                             r"no (?:me )?(?:lo resuelvas|resuelvas|digas|des|desveles))\b", peticion, re.I)
    if resolver and not solo_lectura:
        respuesta = cliente.responses.parse(model=modelo, store=False, max_output_tokens=5600,
            instructions=GUIA_CONSULTA + " Incluye un elemento por cada ejercicio solicitado. En el campo de "
            "resultado escribe todos los valores o la solución simbólica, no instrucciones para obtenerlos. "
            "Una consulta anterior sirve para identificar la referencia; responde a la pregunta actual. "
            "Si un signo es ambiguo, declara la interpretación o da las alternativas. Si solo se solicitan "
            "los resultados, deja vacíos el cálculo y la comprobación.",
            text_format=Soluciones, input=entrada)
        solucion = respuesta.output_parsed
        if not solucion or not solucion.ejercicios:
            raise HTTPException(502, "No se pudo completar la solución de los ejercicios.")
        bloques = []
        for item in solucion.ejercicios:
            if not item.resultado_o_dato_faltante.strip():
                raise HTTPException(502, "Faltó el resultado de un ejercicio.")
            bloques.append("\n".join(filter(None, [item.ejercicio, "Respuesta: " + item.resultado_o_dato_faltante,
                          item.calculo, item.comprobacion])))
        return {"respuesta": "\n\n".join(bloques)}
    respuesta = cliente.responses.create(model=modelo, store=False, max_output_tokens=3200,
                                         instructions=GUIA_CONSULTA, input=entrada)
    if not respuesta.output_text:
        raise HTTPException(502, "El análisis no devolvió texto.")
    return {"respuesta": respuesta.output_text}


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


def marca_de_agua(imagen, texto):
    import numpy as np
    from scipy.ndimage import binary_propagation

    pixeles = np.array(imagen)
    alfa = pixeles[:, :, 3].astype(float)
    if alfa.min() == 255:
        # Retira el fondo claro conectado al borde y protege los blancos interiores del logo.
        claros = pixeles[:, :, :3].min(axis=2) >= 230
        borde = np.zeros_like(claros)
        borde[[0, -1], :] = claros[[0, -1], :]
        borde[:, [0, -1]] = claros[:, [0, -1]]
        alfa[binary_propagation(borde, mask=claros)] = 0
    anterior = max(1, alfa.max())
    fuerza = .64
    if re.search(r"m[aá]s (?:visible|opac[ao]|intens[ao])|menos transparente", texto, re.I):
        fuerza = min(1, anterior / 255 + .18) if anterior < 255 else .82
    elif re.search(r"menos visible|m[aá]s (?:tenue|transparente)", texto, re.I):
        fuerza = max(.12, anterior / 255 - .18) if anterior < 255 else .45
    pixeles[:, :, 3] = np.rint(alfa * (255 * fuerza / anterior)).clip(0, 255).astype("uint8")
    salida = io.BytesIO()
    Image.fromarray(pixeles).save(salida, "PNG")
    return Response(salida.getvalue(), media_type="image/png")


def procesar(tipo, datos, texto, cliente, modelo, modo="editar"):
    if tipo in VIDEOS:
        from video_chat import editar_video
        return editar_video(tipo, datos, texto, cliente, modelo)
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
        if modo == "consultar":
            if cliente is None:
                raise HTTPException(503, "El análisis visual no está configurado.")
            muestra = io.BytesIO()
            imagen.save(muestra, "PNG")
            url = "data:image/png;base64," + base64.b64encode(muestra.getvalue()).decode("ascii")
            return consultar_adjunto(cliente, modelo, {"type":"input_image", "image_url":url, "detail":"high"}, texto)
        if modo == "marca":
            return marca_de_agua(imagen, texto)
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
    return consultar_adjunto(cliente, modelo, contenido, texto)


def generar_diseno(texto, cliente):
    texto = texto.strip()
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




class ControlEdicion(BaseModel):
    id: UUID


class PeticionDiseno(BaseModel):
    id: UUID | None = None
    texto: str = Field(min_length=1, max_length=4000)


def conectar_adjuntos(app, autorizar, cliente, modelo):
    conectar_reels(app, autorizar, cliente, modelo)

    @app.post("/chat/diseno", dependencies=[Depends(autorizar)])
    def diseno(peticion: PeticionDiseno):
        if peticion.id:
            return ejecutar_edicion(peticion.id, "diseno", texto=peticion.texto)
        return generar_diseno(peticion.texto, cliente)

    @app.post("/chat/edicion/cancelar", dependencies=[Depends(autorizar)])
    def detener(peticion: ControlEdicion):
        return cancelar(peticion.id)

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
            if len(datos) + len(trozo) > (36 * 1024 * 1024 if tipo in VIDEOS else MAX_BYTES):
                raise HTTPException(413, "El archivo supera el tamaño admitido.")
            datos.extend(trozo)
        if not datos:
            raise HTTPException(422, "El archivo está vacío.")
        modo = request.headers.get("x-aura-modo", "editar")
        if modo not in {"consultar", "marca", "editar"}:
            raise HTTPException(422, "Modo de adjunto inválido.")
        trabajo = request.headers.get("x-aura-trabajo")
        try:
            if trabajo and (tipo in VIDEOS or tipo.startswith("image/")) and modo != "consultar":
                return await run_in_threadpool(partial(ejecutar_edicion, trabajo, "foto", bytes(datos),
                                                       tipo=tipo, texto=texto, modelo=modelo, modo=modo))
            return await run_in_threadpool(partial(procesar, tipo, bytes(datos), texto, cliente, modelo, modo))
        except HTTPException:
            raise
        except Exception as exc:
            codigo = getattr(exc, "status_code", None)
            if codigo in {400, 403, 429}:
                raise HTTPException(422 if codigo == 400 else 503,
                                    "El proveedor rechazó la solicitud. Revisa el archivo o la disponibilidad del modelo.") from None
            raise HTTPException(502, "El servicio no pudo completar la solicitud. Conserva el adjunto para reintentar.") from None
