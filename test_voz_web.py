import io
import unittest
import wave
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from voz_web import voice_routes


class AudioAPI:
    def __init__(self):
        self.calls = []
        self.audio = SimpleNamespace(transcriptions=self, speech=self)

    def create(self, **options):
        self.calls.append(options)
        return SimpleNamespace(text="Hola, AURA.", content=b"audio-de-prueba")


class VoiceTest(unittest.TestCase):
    def setUp(self):
        def authorize(request: Request):
            if request.headers.get("Authorization") != "Bearer test":
                raise HTTPException(401)
        self.ai = AudioAPI()
        app = FastAPI()
        app.include_router(voice_routes(authorize, lambda: self.ai))
        self.http = TestClient(app)

    def test_turn(self):
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", ""))
            wav.writeframes(b"\0\0" * 1600)
        headers = {"Authorization":"Bearer test", "Content-Type":"audio/wav"}
        response = self.http.post("/voz/transcribir", content=audio.getvalue(), headers=headers)
        self.assertEqual(response.json(), {"texto":"Hola, AURA."})
        self.assertEqual(self.ai.calls[0]["file"][1], audio.getvalue())
        headers["Content-Type"] = "text/plain"
        speech = self.http.post("/voz/hablar", content="Hola.", headers=headers)
        self.assertEqual(speech.content, b"audio-de-prueba")
        self.assertEqual(speech.headers["content-type"], "audio/mpeg")
        self.assertEqual(self.ai.calls[1]["input"], "Hola.")

    def test_invalid_input_does_not_call_provider(self):
        self.assertEqual(self.http.post("/voz/hablar", content="Hola").status_code, 401)
        self.assertEqual(self.http.post("/voz/transcribir", content=b"bad",
            headers={"Authorization":"Bearer test","Content-Type":"audio/wav"}).status_code, 422)
        self.assertEqual(self.ai.calls, [])


if __name__ == "__main__":
    unittest.main()
