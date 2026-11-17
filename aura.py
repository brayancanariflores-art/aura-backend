import os
import json
import urllib.request
import urllib.parse
import re
import ssl
from datetime import datetime, timezone, timedelta
import yfinance as yf
import chromadb
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Aura Cloud Brain", version="1.0")

ai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ssl_context = ssl._create_unverified_context()

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection("Aura_Knowledge_Base")

def inicializar_base_historial():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rol TEXT,
            contenido TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

inicializar_base_historial()

def guardar_en_historial(rol, contenido):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO historial (rol, contenido) VALUES (?, ?)", (rol, contenido))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error guardando en historial: {e}")

def cargar_historial_reciente(limite=10):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT rol, contenido FROM historial ORDER BY id DESC LIMIT ?", (limite,))
        rows = cursor.fetchall()
        conn.close()
        
        historial_cargado = []
        for rol, contenido in reversed(rows):
            historial_cargado.append({"role": rol, "content": contenido})
        return historial_cargado
    except Exception as e:
        print(f"Error cargando historial: {e}")
        return []

def buscar_en_base_vectorial(pregunta_usuario):
    try:
        resultados = collection.query(
            query_texts=[pregunta_usuario],
            n_results=2
        )
        documentos = resultados.get("documents", [[]])[0]
        return "\n".join(documentos)
    except Exception as e:
        return ""

def web_search(query):
    try:
        query_lower = query.lower()
        city = "Elizabeth"
        words = query.split()
        if "en" in words:
            idx = words.index("en")
            if idx + 1 < len(words):
                city = words[idx+1].replace("?", "").replace(".", "")

        if "clima" in query_lower or "tiempo" in query_lower or "temperatura" in query_lower:
            weather_url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
            req_w = urllib.request.Request(weather_url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req_w, context=ssl_context, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                current = data['current_condition'][0]
                res_txt = f"Clima actual en {city}: {current['weatherDesc'][0]['value']}, Temperatura de {current['temp_C']} grados Celsius."
                return json.dumps([{"snippet": res_txt}], ensure_ascii=False)

        data_form = urllib.parse.urlencode({'q': query}).encode('utf-8')
        req_lite = urllib.request.Request(
            "https://lite.duckduckgo.com/lite/",
            data=data_form,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req_lite, context=ssl_context, timeout=12) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            snippets = re.findall(r'class=["\']result-snippet["\'][^>]*>(.*?)<\/td>', html, re.DOTALL)
            if snippets:
                cleaned = [re.sub(r'<.*?>', '', s).strip() for s in snippets[:4]]
                results = [{"snippet": c} for c in cleaned if c]
                if results:
                    return json.dumps(results, ensure_ascii=False)
        return f"Resultados web para {query}: No se encontraron fragmentos."
    except Exception as e:
        return f"Error en búsqueda web: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Busca información en tiempo real en internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Término de búsqueda."}
                },
                "required": ["query"]
            }
        }
    }
]

class PeticionChat(BaseModel):
    texto: str

@app.post("/chat")
def procesar_chat(peticion: PeticionChat):
    try:
        pregunta = peticion.texto
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "Eres Aura, una amiga y asistente personal sumamente cálida, empática y amigable. "
                    "Hablas EXCLUSIVAMENTE en español latino neutro y natural. "
                    "Tus respuestas deben ser cortas y dinámicas. "
                    "Al mencionar números, cantidades, horas o fechas, escríbelos siempre en palabras. "
                    "Estamos en el año dos mil veintiséis."
                )
            }
        ]
        
        messages.extend(cargar_historial_reciente(limite=10))
        
        contexto_memoria = buscar_en_base_vectorial(pregunta)
        prompt = f"Contexto de base de conocimientos: {contexto_memoria}. Mensaje del usuario: {pregunta}"
        
        messages.append({"role": "user", "content": prompt})
        guardar_en_historial("user", prompt)
        
        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            tool_calls_dict = []
            for tc in response_message.tool_calls:
                tool_calls_dict.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })
            
            messages.append({"role": "assistant", "content": response_message.content, "tool_calls": tool_calls_dict})
            
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "web_search":
                    args = json.loads(tool_call.function.arguments)
                    search_result = web_search(args.get("query"))
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": search_result})
            
            second_response = ai_client.chat.completions.create(model="gpt-4o", messages=messages)
            assistant_reply = second_response.choices[0].message.content
        else:
            assistant_reply = response_message.content
            
        guardar_en_historial("assistant", assistant_reply)
        return {"respuesta": assistant_reply}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))