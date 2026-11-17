import json
import logging
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from html import unescape

import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura")

app = FastAPI(title="Aura Cloud Brain", version="1.1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DATA_DIR = os.getenv("AURA_DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_DB = os.path.join(DATA_DIR, "chat_history.db")

# El cliente se crea solo si existe la clave; nunca mostrarla en los logs.
api_key = os.getenv("OPENAI_API_KEY")
ai_client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1) if api_key else None
if ai_client is None:
    logger.warning("OPENAI_API_KEY no esta configurada")

client_db = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
collection = client_db.get_or_create_collection("Aura_Knowledge_Base")


def conectar_historial():
    return sqlite3.connect(HISTORY_DB, timeout=10)


def inicializar_base_historial():
    with conectar_historial() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historial (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rol TEXT NOT NULL,
                contenido TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)


inicializar_base_historial()


def guardar_en_historial(rol, contenido):
    if rol not in ("user", "assistant") or not isinstance(contenido, str):
        return
    try:
        with conectar_historial() as conn:
            conn.execute(
                "INSERT INTO historial (rol, contenido) VALUES (?, ?)",
                (rol, contenido),
            )
    except sqlite3.Error:
        logger.exception("No se pudo guardar el historial")


def cargar_historial_reciente(limite=10):
    try:
        with conectar_historial() as conn:
            rows = conn.execute(
                "SELECT rol, contenido FROM historial "
                "WHERE rol IN ('user', 'assistant') ORDER BY id DESC LIMIT ?",
                (limite,),
            ).fetchall()
        return [{"role": rol, "content": contenido} for rol, contenido in reversed(rows)]
    except sqlite3.Error:
        logger.exception("No se pudo leer el historial")
        return []


def buscar_en_base_vectorial(pregunta_usuario):
    try:
        if collection.count() == 0:
            return ""
        resultados = collection.query(query_texts=[pregunta_usuario], n_results=2)
        documentos = resultados.get("documents") or [[]]
        return "\n".join(str(doc) for doc in documentos[0] if doc)
    except Exception:
        logger.exception("Error consultando la memoria vectorial")
        return ""


def web_search(query):
    if not isinstance(query, str) or not query.strip():
        return "La consulta de busqueda esta vacia."
    try:
        query_lower = query.lower()
        city = "Elizabeth"
        words = query.split()
        words_lower = [word.lower() for word in words]
        if "en" in words_lower:
            idx = words_lower.index("en")
            if idx + 1 < len(words):
                city = words[idx + 1].strip("?.!,")

        if any(term in query_lower for term in ("clima", "tiempo", "temperatura")):
            weather_url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req = urllib.request.Request(weather_url, headers={"User-Agent": "AuraCloudBrain/1.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            current = data["current_condition"][0]
            result = (
                f"Clima actual en {city}: {current['weatherDesc'][0]['value']}, "
                f"temperatura de {current['temp_C']} grados Celsius."
            )
            return json.dumps([{"snippet": result}], ensure_ascii=False)

        form = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://lite.duckduckgo.com/lite/",
            data=form,
