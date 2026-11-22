"""Entrada y salida de voz del chat web. Los fragmentos se procesan en memoria."""
import io
import os
import wave

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

MAX_SECONDS = 90
MAX_WAV = 44 + MAX_SECONDS * 16000 * 2
NO_STORE = {"Cache-Control": "private, no-store"}


async def read_limited(request, maximum):
    data = bytearray()
    async for part in request.stream():
        if len(data) + len(part) > maximum:
            raise HTTPException(413, "Fragmento de voz demasiado largo.")
        data.extend(part)
    if not data:
        raise HTTPException(422, "El fragmento está vacío.")
    return bytes(data)


def voice_routes(authorize, get_client):
    router = APIRouter(prefix="/voz", dependencies=[Depends(authorize)])

    def client():
        ai = get_client()
        if ai is None:
            raise HTTPException(503, "Servicio de voz no disponible.")
        return ai

    @router.post("/transcribir")
    async def transcribe(request: Request, response: Response):
        if request.headers.get("content-type", "").split(";")[0] != "audio/wav":
            raise HTTPException(415, "Se requiere audio WAV.")
        data = await read_limited(request, MAX_WAV)
        try:
            with wave.open(io.BytesIO(data), "rb") as audio:
                frames = audio.getnframes()
                if (audio.getparams()[:3] != (1, 2, 16000) or not 0 < frames <= MAX_SECONDS * 16000
                        or len(audio.readframes(frames)) != frames * 2):
                    raise ValueError
        except (EOFError, wave.Error, ValueError):
            raise HTTPException(422, "Fragmento WAV no válido.") from None
        transcriptions = client().audio.transcriptions
        def recognize():
            options = dict(file=("turno.wav", data, "audio/wav"), language="es")
            model = os.getenv("AURA_STT_MODEL", "gpt-transcribe")
            try:
                return transcriptions.create(model=model, **options)
            except Exception as exc:
                if model != "gpt-transcribe" or getattr(exc, "code", None) != "model_not_found":
                    raise
                return transcriptions.create(model="gpt-4o-mini-transcribe", **options)
        try:
            result = await run_in_threadpool(recognize)
        except Exception:
            raise HTTPException(502, "La transcripción no está disponible.") from None
        response.headers.update(NO_STORE)
        return {"texto": result.text.strip()}

    @router.post("/hablar")
    async def speak(request: Request):
        if request.headers.get("content-type", "").split(";")[0] != "text/plain":
            raise HTTPException(415, "Se requiere texto.")
        data = await read_limited(request, 16000)
        try:
            text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            raise HTTPException(422, "Texto no válido.") from None
        if not 0 < len(text) <= 3500:
            raise HTTPException(422, "Longitud de texto no válida.")
        speech = client().audio.speech
        try:
            result = await run_in_threadpool(
                speech.create, model=os.getenv("AURA_TTS_MODEL", "gpt-4o-mini-tts"),
                voice="marin", input=text, response_format="mp3",
                instructions="Habla en español latinoamericano, con voz cálida y natural. Lee el texto tal como está.",
            )
        except Exception:
            raise HTTPException(502, "La voz no está disponible.") from None
        return Response(result.content, media_type="audio/mpeg", headers=NO_STORE)

    return router
