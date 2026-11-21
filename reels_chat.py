"""Reels privados para el chat. Los archivos de prueba caducan al cabo de una hora."""
import base64
import fcntl
import hashlib
import io
import json
import logging
import re
import shutil
import tempfile
import time
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from estudio_medios import MEDIDAS, reel_narrado
from formato_chat import lienzo_pedido


class PedidoReel(BaseModel):
    texto: str = Field(min_length=1, max_length=4000)
    id: UUID


class ConsultaReel(BaseModel):
    id: UUID


def formato_del_pedido(texto):
    pedido = texto.partition("\nContexto de la conversación:")[0].casefold()
    orientacion = re.search(r"\b(vertical|horizontal|9\s*[:x×]\s*16|16\s*[:x×]\s*9)\b", pedido)
    if orientacion:
        return "horizontal" if orientacion[0].startswith(("horizontal", "16")) else "vertical"
    if re.search(r"\b(?:shorts?|redes sociales)\b", pedido):
        return "vertical"
    destino = re.search(r"\b(?:para|en|formato|destino)\s+(?:(?:mi|el|un|una)\s+)?"
                        r"(?:(?:canal|plataforma|directo)\s+de\s+)?(?:youtube|you tube|twitch|streaming)\b", pedido)
    return "horizontal" if destino else "vertical"


def preparar_reel(carpeta, texto, cliente, modelo):
    formato = formato_del_pedido(texto)
    medidas = lienzo_pedido(texto, video=True) or MEDIDAS[formato]
    plan = cliente.with_options(timeout=35, max_retries=0).chat.completions.create(
        model=modelo, response_format={"type": "json_object"}, messages=[
            {"role": "system", "content": (
                "Realiza un video corto sobre el tema solicitado, respetando el idioma, el estilo y "
                "los nombres. Puede ser ficción, un relato personal, educación, una receta o publicidad. "
                "Aplica tono comercial solo si se pide. Devuelve JSON con idioma_voz e idioma_subtitulos "
                "(códigos ISO 639-2 de tres letras). Si se pide sin subtítulos, idioma_subtitulos es null. "
                "La voz usa el idioma elegido o español por defecto. Los subtítulos usan su propio idioma "
                "elegido o, por defecto, el de la voz. Incluye escenas, exactamente tres objetos con "
                "visual (descripción de una toma), frase (narración breve en el idioma de la voz) y "
                "subtitulo (traducción fiel y breve de la frase al idioma de los subtítulos, o null si se omiten). "
                "Usa la petición actual como prioridad y el contexto solo para resolver referencias. "
                "No inventes precios, direcciones ni testimonios.")},
            {"role": "user", "content": texto}])
    guion = json.loads(plan.choices[0].message.content)
    escenas = guion["escenas"]
    idioma_voz = guion["idioma_voz"]
    idioma_subtitulos = guion["idioma_subtitulos"]
    if not re.fullmatch("[a-z]{3}", idioma_voz) or (
        idioma_subtitulos is not None and not re.fullmatch("[a-z]{3}", idioma_subtitulos)
    ):
        raise ValueError("Idiomas de narración o subtítulos inválidos")
    if not isinstance(escenas, list) or len(escenas) != 3:
        raise ValueError("Guion incompleto")
    campos = [("visual", 1000), ("frase", 150)]
    if idioma_subtitulos:
        campos.append(("subtitulo", 240))
    for escena in escenas:
        for campo, limite in campos:
            valor = escena[campo]
            if not isinstance(valor, str) or not 0 < len(valor.strip()) <= limite:
                raise ValueError("Escena incompleta")
    horizontal = medidas[0] > medidas[1]
    if horizontal:
        lienzo = "1024x1536"
        paneles = "Tres viñetas panorámicas 16:9 de igual altura apiladas de arriba abajo, una por fila."
    else:
        lienzo = "1536x1024"
        paneles = "Tres viñetas verticales 9:16 de igual ancho de izquierda a derecha, una por columna."
    imagen = cliente.with_options(timeout=150, max_retries=0).images.generate(
        model="gpt-image-1.5", size=lienzo, quality="medium", output_format="jpeg", n=1,
        prompt=(paneles + f" Cada panel se reencuadrará a {medidas[0]} por {medidas[1]}; centra el motivo. "
                "Llena todo el lienzo, sin márgenes, separadores ni rótulos. "
                "Mantén la continuidad visual, el estilo y los personajes entre escenas. "
                + "\n".join(f"Panel {i+1}: {e['visual']}" for i, e in enumerate(escenas))))
    with Image.open(io.BytesIO(base64.b64decode(imagen.data[0].b64_json, validate=True))) as tira:
        if (tira.height > tira.width) != horizontal or tira.width == tira.height or tira.width * tira.height > 12000000:
            raise ValueError("Encuadre inesperado")
        tomas = []
        for i in range(3):
            toma = carpeta / f"escena-{i}.jpg"
            ancho, alto = tira.size
            if horizontal:
                caja = (0, i * alto // 3, ancho, (i + 1) * alto // 3)
            else:
                caja = (i * ancho // 3, 0, (i + 1) * ancho // 3, alto)
            tira.crop(caja).convert("RGB").save(toma)
            tomas.append(toma)

    def narrar(frase):
        return cliente.with_options(timeout=35, max_retries=0).audio.speech.create(
            model="gpt-4o-mini-tts", voice="coral", response_format="pcm", input=frase).content

    subtitulos = [e["subtitulo"] for e in escenas] if idioma_subtitulos else []
    return reel_narrado(tomas[0], carpeta, {"tipo": "foto"}, {}, formato,
                        [e["frase"] for e in escenas], narrar, escenas=tomas, medidas=medidas,
                        subtitulos=subtitulos, idioma_voz=idioma_voz, idioma_subtitulos=idioma_subtitulos)[0]


def conectar_reels(app, autorizar, cliente, modelo):
    raiz = Path(tempfile.gettempdir()) / "aura-chat-reels"
    raiz.mkdir(mode=0o700, exist_ok=True)
    privado = {"Cache-Control": "private, no-store"}

    def ejecutar_trabajo(carpeta, texto):
        try:
            archivo = preparar_reel(carpeta, texto, cliente, modelo)
            archivo.replace(carpeta / "listo.mp4")
        except Exception as exc:
            logging.getLogger(__name__).warning("Reel fallido (%s)", type(exc).__name__)
            (carpeta / "error").touch()
        finally:
            for ruta in carpeta.iterdir():
                if ruta.name not in {"pedido", "listo.mp4", "error"}:
                    ruta.unlink(missing_ok=True)

    @app.post("/chat/reel", dependencies=[Depends(autorizar)], status_code=202)
    def crear(pedido: PedidoReel, tareas: BackgroundTasks):
        if not cliente:
            raise HTTPException(503, "Configura el generador de AURA.")
        texto = pedido.texto.strip()
        if not texto:
            raise HTTPException(422, "Describe el video.")
        try:
            lienzo_pedido(texto, video=True)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        carpeta = raiz / str(pedido.id)
        huella = hashlib.sha256(texto.encode()).hexdigest()
        with (raiz / ".cola").open("a") as bloqueo:
            fcntl.flock(bloqueo, fcntl.LOCK_EX)
            activos = 0
            for item in raiz.iterdir():
                if not item.is_dir():
                    continue
                edad = time.time() - item.stat().st_mtime
                if edad > 3600:
                    shutil.rmtree(item)
                elif not (item / "listo.mp4").is_file() and not (item / "error").exists():
                    if edad > 600:
                        (item / "error").touch()
                    else:
                        activos += 1
            if carpeta.exists():
                if (carpeta / "pedido").read_text() != huella:
                    raise HTTPException(409, "Este identificador corresponde a otro pedido.")
            elif activos >= 2:
                raise HTTPException(429, "AURA ya está preparando dos videos.")
            else:
                carpeta.mkdir(mode=0o700)
                (carpeta / "pedido").write_text(huella)
                tareas.add_task(ejecutar_trabajo, carpeta, texto)
        return JSONResponse({"id": str(pedido.id)}, status_code=202, headers=privado)

    @app.post("/chat/reel/estado", dependencies=[Depends(autorizar)])
    def estado(pedido: ConsultaReel):
        carpeta = raiz / str(pedido.id)
        if not carpeta.is_dir() or time.time() - carpeta.stat().st_mtime > 3600:
            raise HTTPException(404, "El reel ya no está disponible.")
        if (carpeta / "listo.mp4").is_file():
            return FileResponse(carpeta / "listo.mp4", media_type="video/mp4", headers=privado)
        if (carpeta / "error").exists() or time.time() - carpeta.stat().st_mtime > 600:
            raise HTTPException(502, "No se pudo completar el reel.")
        return JSONResponse({"estado": "preparando"}, status_code=202, headers=privado)
