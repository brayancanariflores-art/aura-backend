"""Reels privados para el chat. Los archivos de prueba caducan al cabo de una hora."""
import base64
import fcntl
import hashlib
import io
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from estudio_medios import reel_narrado


class PedidoReel(BaseModel):
    texto: str = Field(min_length=1, max_length=4000)
    id: UUID


class ConsultaReel(BaseModel):
    id: UUID


def preparar_reel(carpeta, texto, cliente, modelo):
    plan = cliente.with_options(timeout=35, max_retries=0).chat.completions.create(
        model=modelo, response_format={"type": "json_object"}, messages=[
            {"role": "system", "content": (
                "Realiza un video corto sobre el tema solicitado, respetando el idioma, el estilo y "
                "los nombres. Puede ser ficción, un relato personal, educación, una receta o publicidad. "
                "Aplica tono comercial solo si se pide. Devuelve JSON con escenas: tres objetos con "
                "visual (descripción de una toma) y frase (una frase breve para voz y subtítulo). "
                "Usa la petición actual como prioridad y el contexto solo para resolver referencias. "
                "No inventes precios, direcciones ni testimonios.")},
            {"role": "user", "content": texto}])
    escenas = json.loads(plan.choices[0].message.content)["escenas"]
    if not isinstance(escenas, list) or len(escenas) != 3:
        raise ValueError("Guion incompleto")
    for escena in escenas:
        for campo, limite in (("visual", 1000), ("frase", 150)):
            valor = escena[campo]
            if not isinstance(valor, str) or not 0 < len(valor.strip()) <= limite:
                raise ValueError("Escena incompleta")
    imagen = cliente.with_options(timeout=150, max_retries=0).images.generate(
        model="gpt-image-1.5", size="1536x1024", quality="medium", output_format="jpeg", n=1,
        prompt=("Un tríptico horizontal, exactamente tres paneles verticales del mismo ancho, "
                "sin márgenes ni separadores. Cada panel llena toda la altura. Sin rótulos. "
                "Mantén la continuidad visual, el estilo y los personajes entre escenas. "
                + "\n".join(f"Panel {i+1}: {e['visual']}" for i, e in enumerate(escenas))))
    with Image.open(io.BytesIO(base64.b64decode(imagen.data[0].b64_json, validate=True))) as tira:
        if tira.width < tira.height or tira.width * tira.height > 12000000:
            raise ValueError("Encuadre inesperado")
        tomas = []
        for i in range(3):
            toma = carpeta / f"escena-{i}.jpg"
            tira.crop((i*tira.width//3, 0, (i+1)*tira.width//3, tira.height)).convert("RGB").save(toma)
            tomas.append(toma)

    def narrar(frase):
        return cliente.with_options(timeout=35, max_retries=0).audio.speech.create(
            model="gpt-4o-mini-tts", voice="coral", response_format="pcm", input=frase).content

    return reel_narrado(tomas[0], carpeta, {"tipo": "foto"}, {}, "vertical",
                        [e["frase"] for e in escenas], narrar, escenas=tomas)[0]


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
