import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from estudio_medios import ejecutar, entrada_video
from reels_chat import conectar_reels, huella_medios


class RevisionReelTest(unittest.TestCase):
    def setUp(self):
        directorio = tempfile.TemporaryDirectory()
        self.addCleanup(directorio.cleanup)
        self.raiz = Path(directorio.name)
        self.video = self.raiz / "muestra.mp4"
        ejecutar(["-v", "error", "-f", "lavfi", "-i", "color=s=80x80:r=4:d=1",
                  "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264",
                  "-c:a", "aac", str(self.video)])

    def test_rechaza_duplicado_en_reel_sin_marca(self):
        app = FastAPI()
        with patch("reels_chat.tempfile.gettempdir", return_value=str(self.raiz)):
            conectar_reels(app, lambda: None, object(), "prueba")

        def renderizar(carpeta, texto, cliente, modelo):
            (carpeta / "version.json").write_text(json.dumps({"guion": {"escenas": []}}))
            (carpeta / "escenas.jpg").touch()
            return Path(shutil.copyfile(self.video, carpeta / "salida.mp4"))

        with patch("reels_chat.preparar_reel", side_effect=renderizar), TestClient(app) as chat:
            primero, segundo = str(uuid4()), str(uuid4())
            for pedido in ({"texto": "Crea un reel vertical", "id": primero},
                           {"texto": "Cambia el fondo", "id": segundo, "anterior": primero}):
                self.assertEqual(chat.post("/chat/reel", json=pedido).status_code, 202)
            revision = chat.post("/chat/reel/estado", json={"id": segundo})
            self.assertEqual(revision.status_code, 422)
            self.assertEqual(revision.headers.get("X-Aura-Edicion"), "ajuste-no-aplicado")
            anterior = chat.post("/chat/reel/estado", json={"id": primero})
            self.assertEqual(anterior.content, self.video.read_bytes())

    def test_subtitulos_y_audio_cuentan_como_cambios(self):
        srt = self.raiz / "subtitulos.srt"
        srt.write_text("1\n00:00:00,100 --> 00:00:00,850\nTexto revisado\n")
        titulado = self.raiz / "titulado.mp4"
        ejecutar(["-v", "error", *entrada_video(self.video), "-i", str(srt), "-map", "0",
                  "-map", "1:s", "-c", "copy", "-c:s", "mov_text", str(titulado)])
        mudo = self.raiz / "mudo.mp4"
        ejecutar(["-v", "error", *entrada_video(self.video), "-an", "-c", "copy", str(mudo)])
        base = huella_medios(self.video)
        self.assertNotEqual(base, huella_medios(titulado))
        self.assertNotEqual(base, huella_medios(mudo))

    def test_ignora_metadatos_del_contenedor(self):
        copia = self.raiz / "otro-titulo.mp4"
        ejecutar(["-v", "error", *entrada_video(self.video), "-c", "copy",
                  "-metadata", "title=Otro titulo de archivo", str(copia)])
        self.assertNotEqual(self.video.read_bytes(), copia.read_bytes())
        self.assertEqual(huella_medios(self.video), huella_medios(copia))
