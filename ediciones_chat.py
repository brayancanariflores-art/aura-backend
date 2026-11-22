"""Trabajos privados de edición. Cada proceso y sus hijos comparten un grupo cancelable."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from fastapi.responses import Response


def identificador(valor):
    try:
        return str(UUID(str(valor)))
    except ValueError:
        raise HTTPException(422, "Identificador de edición inválido.") from None


def control(valor):
    raiz = Path(tempfile.gettempdir()) / "aura-control-ediciones"
    raiz.mkdir(mode=0o700, exist_ok=True)
    # Los marcadores también cubren una cancelación que llega antes del archivo.
    ahora = time.time()
    for archivo in raiz.iterdir():
        try:
            if archivo.stat().st_mtime < ahora - 3600:
                archivo.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
    return raiz / identificador(valor)


def cancelar(valor):
    ruta = control(valor)
    ruta.with_suffix(".parar").touch()
    with ruta.open("a") as bloqueo:
        try:
            fcntl.flock(bloqueo, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"estado": "cancelando"}
        return {"estado": "cancelada"}


def ejecutar_edicion(valor, accion, datos=b"", **parametros):
    ruta = control(valor)
    with ruta.open("a") as bloqueo, tempfile.TemporaryDirectory(prefix="aura-edicion-") as temporal:
        fcntl.flock(bloqueo, fcntl.LOCK_EX)
        detener = ruta.with_suffix(".parar")
        if detener.exists():
            raise HTTPException(409, "Edición cancelada.")
        carpeta = Path(temporal)
        (carpeta / "pedido").write_text(json.dumps(parametros))
        (carpeta / "entrada").write_bytes(datos)
        proceso = subprocess.Popen([sys.executable, __file__, accion, temporal], start_new_session=True,
                                   env={**os.environ, "TMPDIR": temporal})
        try:
            limite = time.monotonic() + (600 if accion == "reel" else 160)
            while proceso.poll() is None:
                if detener.exists():
                    raise HTTPException(409, "Edición cancelada.")
                if time.monotonic() >= limite:
                    raise HTTPException(504, "La edición superó el tiempo de espera.")
                time.sleep(.1)
            if detener.exists():
                raise HTTPException(409, "Edición cancelada.")
            if proceso.returncode or not (carpeta / "estado").is_file():
                raise HTTPException(502, "El editor se interrumpió.")
            estado = json.loads((carpeta / "estado").read_text())
            if "error" in estado:
                raise HTTPException(estado["codigo"], estado["error"])
            return Response((carpeta / "salida").read_bytes(), media_type=estado["tipo"],
                            headers={"Cache-Control": "private, no-store"})
        finally:
            if proceso.poll() is None:
                try:
                    os.killpg(proceso.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proceso.wait()


def trabajador(accion, carpeta):
    parametros = json.loads((carpeta / "pedido").read_text())
    clave = os.getenv("OPENAI_API_KEY", "").strip()
    cliente = None
    if clave:
        from openai import OpenAI
        cliente = OpenAI(api_key=clave, timeout=30.0, max_retries=1)
    try:
        if accion == "reel":
            from reels_chat import renderizar_reel
            renderizar_reel(Path(parametros["destino"]), parametros["texto"], cliente, parametros["modelo"])
            salida = Response(b"")
        else:
            from adjuntos import generar_diseno, procesar
            if accion == "diseno":
                salida = generar_diseno(parametros["texto"], cliente)
            else:
                salida = procesar(datos=(carpeta / "entrada").read_bytes(), cliente=cliente, **parametros)
        (carpeta / "salida").write_bytes(salida.body)
        estado = {"tipo": salida.media_type or "application/octet-stream"}
    except HTTPException as error:
        estado = {"error": error.detail, "codigo": error.status_code}
    except Exception as error:
        estado = {"error": "El editor no pudo completar la solicitud.",
                  "codigo": 422 if getattr(error, "status_code", 0) == 400 else 502}
    (carpeta / "estado").write_text(json.dumps(estado))


if __name__ == "__main__":
    trabajador(sys.argv[1], Path(sys.argv[2]))
