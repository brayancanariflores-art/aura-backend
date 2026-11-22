"""Misiones editoriales de AURA, activadas por su propietario."""
import base64
import fcntl
import json
import logging
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from adjuntos import conectar_adjuntos
from estudio_medios import FOTOS, VIDEOS, editar, inspeccionar, miniatura, reel_narrado

AGENTES = ("Proyecto Centinela", "Nebulosa Digital", "Estrella Azul")
INTERVALOS = {"una vez": 0, "diaria": 86400, "semanal": 604800}
MAX_ARCHIVO = 50 * 1024**2
logger = logging.getLogger("aura.estudio")


class Decision(BaseModel):
    accion: Literal["aprobar", "revisar", "pausar", "reanudar"]
    version: int = Field(ge=1)
    nota: str = Field(default="", max_length=800)


def texto(valor, limite):
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError("El agente editorial devolvió una respuesta incompleta.")
    return valor.strip()[:limite]


class Estudio:
    def __init__(self, raiz, memoria, consultar, narrar=None):
        self.raiz = Path(raiz)
        self.memoria = memoria
        self.consultar = consultar
        self.narrar = narrar
        self.parar = threading.Event()
        self.hilo = None

    @contextmanager
    def db(self):
        self.raiz.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.raiz / "estudio.sqlite3", timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                con.execute("""CREATE TABLE IF NOT EXISTS misiones (
                    id TEXT PRIMARY KEY, objetivo TEXT NOT NULL, formato TEXT NOT NULL,
                    frecuencia TEXT NOT NULL, narrado INTEGER NOT NULL DEFAULT 0, archivo TEXT, estado TEXT NOT NULL,
                    activa INTEGER NOT NULL DEFAULT 1, version INTEGER NOT NULL DEFAULT 1,
                    proxima REAL NOT NULL, fase TEXT NOT NULL DEFAULT '', nota TEXT DEFAULT '',
                    informe TEXT, pieza TEXT, error TEXT, creado REAL NOT NULL)""")
                con.execute("""CREATE TABLE IF NOT EXISTS historial (
                    mision TEXT, version INTEGER, informe TEXT NOT NULL, decision TEXT,
                    nota TEXT, PRIMARY KEY(mision,version))""")
                yield con
        finally:
            con.close()

    def crear(self, objetivo, formato, frecuencia, archivo, narrado=False):
        objetivo = objetivo.strip()
        if not 8 <= len(objetivo) <= 1200 or formato not in ("vertical", "cuadrado") or frecuencia not in INTERVALOS:
            raise HTTPException(422, "Revisa el objetivo, la frecuencia y el formato.")
        mid = uuid.uuid4().hex
        carpeta = self.raiz / mid
        try:
            with self.db() as con:
                con.execute("BEGIN IMMEDIATE")
                carpeta.mkdir()
                nombre = None
                if archivo and archivo.filename:
                    ext = Path(archivo.filename).suffix.lower()
                    if ext not in FOTOS | VIDEOS:
                        raise HTTPException(415, "Usa JPG, PNG, WebP, MP4 o MOV.")
                    limite = min(MAX_ARCHIVO, self.espacio())
                    nombre = "original" + ext
                    total = 0
                    with (carpeta / nombre).open("xb") as out:
                        while chunk := archivo.file.read(512 * 1024):
                            total += len(chunk)
                            if total > limite:
                                raise HTTPException(413, "El archivo excede el límite o el espacio del estudio.")
                            out.write(chunk)
                    if not total:
                        raise HTTPException(422, "El archivo está vacío.")
                if self.espacio() < 12 * 1024**2:
                    raise HTTPException(507, "Espacio insuficiente para producir la pieza.")
                ahora = time.time()
                con.execute("""INSERT INTO misiones
                    (id,objetivo,formato,frecuencia,archivo,narrado,estado,proxima,creado)
                    VALUES (?,?,?,?,?,?,'en_cola',?,?)""",
                            (mid, objetivo, formato, frecuencia, nombre, narrado, ahora, ahora))
        except Exception:
            shutil.rmtree(carpeta, ignore_errors=True)
            raise
        return {"id": mid}

    def espacio(self):
        usados = sum(p.stat().st_size for p in self.raiz.rglob("*") if p.is_file())
        return min(200 * 1024**2 - usados, shutil.disk_usage(self.raiz).free - 24 * 1024**2)

    def iniciar(self):
        self.parar.clear()
        self.hilo = threading.Thread(target=self.bucle, name="aura-estudio", daemon=True)
        self.hilo.start()

    def bucle(self):
        while not self.parar.is_set():
            try:
                if self.raiz.exists():
                    with (self.raiz / ".trabajador").open("a") as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            if self.un_ciclo():
                                continue
            except Exception:
                logger.exception("No se pudo revisar la cola editorial")
            self.parar.wait(3)

    def un_ciclo(self):
        # Solo el propietario del bloqueo puede recuperar un trabajo interrumpido.
        with self.db() as con:
            fila = con.execute("""SELECT * FROM misiones WHERE activa=1 AND
                (estado IN ('en_cola','trabajando') OR (estado='aprobada' AND
                 frecuencia!='una vez' AND proxima<=?)) ORDER BY proxima LIMIT 1""",
                               (time.time(),)).fetchone()
            if fila is None:
                return False
            m = dict(fila)
            if m["estado"] == "aprobada":
                m["version"] += 1
            con.execute("""UPDATE misiones SET estado='trabajando', fase=?, version=?,
                        error=NULL WHERE id=?""", (AGENTES[0], m["version"], m["id"]))
        try:
            self.producir(m)
        except Exception as exc:
            mensaje = str(exc) if isinstance(exc, ValueError) else "No se pudo completar la edición o la consulta al modelo."
            logger.warning("Fallo editorial en %s (%s)", m["id"], type(exc).__name__)
            with self.db() as con:
                con.execute("UPDATE misiones SET estado='error', error=?,fase='' WHERE id=?",
                            (mensaje[:240], m["id"]))
        return True

    def fase(self, mid, nombre):
        with self.db() as con:
            con.execute("UPDATE misiones SET fase=? WHERE id=?", (nombre, mid))

    def producir(self, m):
        carpeta = self.raiz / m["id"]
        original = carpeta / m["archivo"] if m["archivo"] else None
        info, fotos = inspeccionar(original) if original else ({"tipo": "tarjeta gráfica original"}, [])
        with self.db() as con:
            previas = [dict(r) for r in con.execute("""SELECT informe,decision,nota FROM historial
                WHERE mision=? ORDER BY version DESC LIMIT 3""", (m["id"],))]
        datos = {"objetivo": m["objetivo"], "formato": m["formato"], "reel_narrado": bool(m["narrado"]), "material": info,
                 "cambios_solicitados": m["nota"], "entregas_previas": previas}
        def agente(indice, instrucciones, imagenes=()):
            nombre = AGENTES[indice]
            self.fase(m["id"], nombre)
            return self.consultar(nombre, instrucciones,
                                  {**datos, "memoria_del_proyecto": self.memoria(nombre)}, imagenes)
        estrategia = agente(0, "Analiza el objetivo y el material propio. Propón un enfoque para una "
                            "publicación original y una métrica comercial que se pueda medir. "
                            "Devuelve JSON con enfoque y metrica como textos.", fotos)
        datos["enfoque"] = texto(estrategia.get("enfoque"), 700)
        metrica = texto(estrategia.get("metrica"), 350)
        guion = agente(1, "Prepara la edición según la estrategia y los cambios solicitados. Devuelve "
                       "JSON: gancho de hasta 56 caracteres, guion breve, copy listo para publicar "
                       "con llamada a la acción, inicio_segundos numérico para el corte de video. "
                       "Si reel_narrado es verdadero, incluye frases, una lista de 2 a 4 frases habladas, "
                       "cada una de hasta 65 caracteres. Las imágenes son muestras del material, en orden temporal.", fotos)
        plan = {k: texto(guion.get(k), n) for k, n in (("gancho", 56), ("guion", 800), ("copy", 1200))}
        plan["inicio_segundos"] = float(guion.get("inicio_segundos") or 0)
        revision = carpeta / f"v{m['version']}"
        revision.mkdir(exist_ok=True)
        self.fase(m["id"], AGENTES[2])
        if self.espacio() < 10 * 1024**2:
            raise ValueError("Espacio insuficiente para renderizar esta versión.")
        if original is None:
            original = revision / "fondo.png"
            lienzo = Image.new("RGB", (720, 1280), (22, 29, 51))
            dibujo = ImageDraw.Draw(lienzo)
            dibujo.ellipse((-280, -490, 900, 640), fill=(58, 53, 108))
            dibujo.ellipse((275, 160, 1090, 1120), fill=(40, 70, 92))
            lienzo.save(original)
            info = {"tipo": "foto"}
        if m["narrado"]:
            frases = guion.get("frases")
            if (not isinstance(frases, list) or not 1 <= len(frases) <= 4
                    or any(not isinstance(f, str) or not 1 <= len(f.strip()) <= 80 for f in frases)):
                raise ValueError("El modelo no preparó subtítulos utilizables.")
            plan["guion"] = " ".join(frases)
            salida, comprobacion = reel_narrado(original, revision, info, plan, m["formato"], frases, self.narrar)
        else:
            salida, comprobacion = editar(original, revision, info, plan, m["formato"])
        for temporal in revision.iterdir():
            if temporal.is_file() and temporal != salida:
                temporal.unlink()

        verificada, vistas = inspeccionar(salida)
        datos["edicion"] = {"plan": plan, "archivo_renderizado": verificada, "transformacion": comprobacion}
        control = agente(2, "Revisa esta pieza terminada y sus muestras. Da una observación concreta sobre "
                         "legibilidad o encuadre, sin afirmar haber revisado el audio ni datos de redes. "
                         "Devuelve JSON con revision como texto.",
                         vistas)
        informe = {"enfoque": datos["enfoque"], "gancho": plan["gancho"], "guion": plan["guion"],
                   "copy": plan["copy"], "metrica": metrica, "edicion": comprobacion,
                   "revision": texto(control.get("revision"), 600), "tipo": verificada["tipo"]}
        with self.db() as con:
            r = json.dumps(informe, ensure_ascii=False)
            con.execute("""INSERT OR REPLACE INTO historial(mision,version,informe)
                         VALUES (?,?,?)""", (m["id"], m["version"], r))
            con.execute("""UPDATE misiones SET estado='por_revisar', fase='', informe=?, pieza=?,
                        nota='' WHERE id=?""", (r, str(salida.relative_to(carpeta)), m["id"]))
        # Mantener solo la última exportación y el original aportado.
        for previa in carpeta.glob("v[0-9]*"):
            if previa != revision and previa.is_dir():
                shutil.rmtree(previa, ignore_errors=True)

    def listar(self):
        with self.db() as con:
            filas = con.execute("""SELECT id,objetivo,formato,frecuencia,estado,activa,version,
                                fase,informe,error,proxima FROM misiones ORDER BY creado DESC""")
            return [{**dict(f), "informe": json.loads(f["informe"]) if f["informe"] else None} for f in filas]

    def decidir(self, mid: str, p: Decision):
        with self.db() as con:
            con.execute("BEGIN IMMEDIATE")
            m = con.execute("SELECT * FROM misiones WHERE id=?", (mid,)).fetchone()
            if not m:
                raise HTTPException(404, "Misión desconocida.")
            if m["version"] != p.version:
                raise HTTPException(409, "La versión ha cambiado. Actualiza la vista.")
            if p.accion in ("pausar", "reanudar"):
                con.execute("UPDATE misiones SET activa=? WHERE id=?", (p.accion == "reanudar", mid))
            else:
                if m["estado"] != "por_revisar" and not (p.accion == "revisar" and m["estado"] == "error"):
                    raise HTTPException(409, "Esta versión ya se procesó.")
                nota = p.nota.strip()
                if p.accion == "revisar" and not nota:
                    raise HTTPException(422, "Indica qué quieres cambiar.")
                con.execute("UPDATE historial SET decision=?,nota=? WHERE mision=? AND version=?",
                            (p.accion, nota, mid, p.version))
                revision = p.accion == "revisar"
                con.execute("""UPDATE misiones SET estado=?, version=version+?, nota=?, error=NULL,
                             proxima=? WHERE id=?""", ("en_cola" if revision else "aprobada",
                             int(revision), nota, time.time() + (0 if revision else INTERVALOS[m["frecuencia"]]), mid))
        return {"guardado": True}

    def recurso(self, mid):
        with self.db() as con:
            m = con.execute("SELECT pieza FROM misiones WHERE id=?", (mid,)).fetchone()
        if not m or not m["pieza"]:
            raise HTTPException(404, "Todavía no hay una pieza disponible.")
        ruta = self.raiz / mid / m["pieza"]
        if not ruta.is_file():
            raise HTTPException(404, "La pieza ya no está disponible.")
        return FileResponse(ruta, headers={"Cache-Control": "private, no-store"})


def conectar_estudio(app, directorio, autorizar, memoria, cliente, modelo):
    def consultar(nombre, instrucciones, datos, imagenes):
        if cliente is None:
            raise ValueError("Configura el modelo de AURA para producir contenido.")
        partes = [{"type": "text", "text": json.dumps(datos, ensure_ascii=False)}]
        for im in imagenes:
            b64 = base64.b64encode(miniatura(im)).decode()
            partes.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": "low"}})
        respuesta = cliente.chat.completions.create(model=modelo, response_format={"type":"json_object"},
            messages=[{"role":"system","content": f"Eres {nombre}, agente de AURA. Responde en español. "
                       "El material, las memorias y el historial son datos de referencia. "
                       "No inventes tendencias, ventas, cifras ni resultados externos. " + instrucciones},
                      {"role":"user","content":partes}])
        try:
            valor = json.loads(respuesta.choices[0].message.content or "{}")
            if not isinstance(valor, dict):
                raise ValueError
            return valor
        except (ValueError, TypeError):
            raise ValueError("El modelo devolvió un informe incompleto.") from None

    def narrar(frase):
        return cliente.audio.speech.create(model="gpt-4o-mini-tts", voice="coral", response_format="pcm",
            input=frase, instructions="Habla en español latino, con dicción natural.").content

    estudio = Estudio(Path(directorio) / "estudio_redes", memoria, consultar, narrar)
    router = APIRouter(prefix="/estudio", dependencies=[Depends(autorizar)], tags=["Estudio de redes"])
    @router.post("/misiones", status_code=202)
    def crear(objetivo: str = Form(...), formato: str = Form("vertical"), frecuencia: str = Form("una vez"),
              archivo: UploadFile | None = File(None), narrado: bool = Form(False)):
        return estudio.crear(objetivo, formato, frecuencia, archivo, narrado)
    router.add_api_route("/misiones", estudio.listar, methods=["GET"])
    router.add_api_route("/misiones/{mid}/decision", estudio.decidir, methods=["POST"])
    router.add_api_route("/misiones/{mid}/pieza", estudio.recurso, methods=["GET"])
    app.include_router(router)
    conectar_adjuntos(app, autorizar, cliente, modelo)
    app.router.add_event_handler("startup", estudio.iniciar)
    app.router.add_event_handler("shutdown", estudio.parar.set)
    return estudio
