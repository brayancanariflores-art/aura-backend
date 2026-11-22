"""Dirección creativa y exportación de una campaña desde un pedido breve de la web."""
import base64
import json
import os
from io import BytesIO
from pathlib import Path
import re

import httpx
import numpy as np
from PIL import Image, ImageOps

from ajustes_marca import AjusteNoAplicado
from estudio_medios import ejecutar, entrada_video, inspeccionar
from control_campana import imagen_de_marca, revisar_accion, revisar_identidad
from movimiento_marca import recortar_logo, renderizar
from tamano_marca import normalizar


def pide_campana(texto):
    pedido = normalizar(texto.partition("\nContexto de la conversación:")[0])
    return bool(re.search(r"\b(?:reel|video)\s+profesional\b.{0,95}\b(?:marca|logo)\b|\b(?:publicitari\w*|publicidad|anuncio|spot|promoci[oó]n|promocional|comercial|cinematografic\w*|fantastic\w*|epic\w*)\b", pedido))


def logo_guardado(texto):
    directorio = os.environ.get("AURA_MARCAS_DIR")
    if not directorio or not Path(directorio).is_dir():
        return None
    palabras = " " + " ".join(re.findall(r"\w+", normalizar(texto))) + " "
    opciones = []
    for archivo in Path(directorio).iterdir():
        if archivo.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp") or not archivo.is_file():
            continue
        marca = " ".join(re.findall(r"[^\W_]+", normalizar(archivo.stem)))
        if marca and f" {marca} " in palabras:
            opciones.append((len(marca), archivo))
    return max(opciones, key=lambda par: par[0])[1] if opciones else None


def dirigir(texto, cliente, modelo, anterior, logo=None):
    if cliente is None:
        orden = normalizar(texto)
        base = anterior or {}
        paleta = next((c for c in ("cobre", "aurora", "ambar") if c in orden), base.get("paleta", "ambar"))
        ritmo = "sereno" if re.search(r"tranquil|seren|lento", orden) else base.get("ritmo", "ascendente")
        return {"paleta":paleta, "ritmo":ritmo, "escena":texto}
    consigna = ("Dirige un anuncio audiovisual desde una sola orden. Inventa una idea visual propia para la marca. "
        "Escribe un tratamiento realizable en 8 segundos, con acción que evoluciona dentro del plano, un movimiento "
        "de cámara motivado, transformación visual, diseño sonoro y un cierre que conserve el movimiento. "
        "Respeta el idioma y los datos aportados. Deja espacio central en el cierre para componer el logo oficial. "
        "No inventes ofertas, premios ni eslóganes atribuidos a la marca. "
        "La animación de marca disponible dibuja un paisaje, estelas de luz, un logo y música. "
        "Marca requiere_generador=true solo si el usuario exige acción, sujetos, diálogos o efectos específicos que ese formato no puede representar. "
        'Devuelve JSON con escena (tratamiento completo), paleta (ambar, cobre o aurora), ritmo (sereno o ascendente) y requiere_generador (booleano).')
    pedido_creativo = [{"type": "text", "text": json.dumps({"pedido": texto, "direccion_anterior": anterior}, ensure_ascii=False)}]
    if logo:
        consigna += (" Lee el logo oficial adjunto antes de elegir la historia. Usa el giro de negocio identificable "
                     "en ese arte. Si no se distingue, usa los rasgos visuales del logo sin atribuirle un negocio por el nombre.")
        pedido_creativo.append(imagen_de_marca(logo))
    resp=cliente.with_options(timeout=35,max_retries=0).chat.completions.create(model=modelo,
        response_format={"type":"json_object"},messages=[{"role":"system","content":consigna},
        {"role":"user","content":pedido_creativo}])
    datos=json.loads(resp.choices[0].message.content)
    if not isinstance(datos.get("escena"),str) or not datos["escena"].strip():
        raise AjusteNoAplicado("La dirección creativa quedó incompleta. Vuelve a intentar este pedido.")
    direccion = {k:str(datos.get(k,""))[:3500] for k in ("escena","paleta","ritmo")}
    direccion["requiere_generador"] = datos.get("requiere_generador") is True
    return direccion


def generar_toma(carpeta, direccion, formato, logo, silencio, historia):
    entrada=[{"type":"text","text":direccion["escena"] + " Ocho segundos. Mantén la acción hasta el último cuadro. "
              "El editor compone el logo oficial al final. Evita dibujar rótulos o duplicar ese logo."}]
    if logo:
        mini=recortar_logo(logo)
        mini.thumbnail((384,384))
        buf=BytesIO()
        mini.save(buf,format="PNG")
        entrada.insert(0,{"type":"image","mime_type":"image/png","data":base64.b64encode(buf.getvalue()).decode()})
    solicitud={"model":"gemini-omni-1.2-flash","input":entrada,
               "response_format":{"type":"video","aspect_ratio":"16:9" if formato=="horizontal" else "9:16"}}
    if historia:
        solicitud["previous_interaction_id"]=historia
    try:
        resp=httpx.post("https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={"x-goog-api-key":os.environ["GEMINI_API_KEY"]}, json=solicitud,timeout=240)
        resp.raise_for_status()
        datos=resp.json()
        video=next(c for paso in datos.get("steps",[]) if paso.get("type")=="model_output"
                   for c in paso.get("content",[]) if c.get("type")=="video" and c.get("mime_type")=="video/mp4")
        contenido=video.get("data","")
        if not 0 < len(contenido) <= 48*1024*1024:
            raise ValueError
        fuente=carpeta/"toma.mp4"
        fuente.write_bytes(base64.b64decode(contenido,validate=True))
    except (httpx.HTTPError,ValueError,StopIteration,KeyError) as e:
        raise AjusteNoAplicado("El generador de movimiento no entregó una toma utilizable para este pedido.") from e
    info,_=inspeccionar(fuente)
    duracion=info["duracion"]
    if not 2<=duracion<=16:
        raise AjusteNoAplicado("La duración de la toma está fuera del intervalo de una campaña corta.")
    revisar_accion(fuente)
    filtros=["[0:v]setsar=1[base]"]
    if formato=="cuadrado":
        filtros=["[0:v]crop='min(iw,ih)':'min(iw,ih)',setsar=1[base]"]
    entradas=entrada_video(fuente)
    pista="[base]"
    if logo:
        marca=recortar_logo(logo)
        dimension=round(min(info["ancho"],info["alto"])*.38)
        marca.thumbnail((dimension,dimension),Image.Resampling.LANCZOS)
        marca.save(carpeta/"cierre.png")
        entradas+=["-loop","1","-framerate","24","-i",str(carpeta/"cierre.png")]
        inicio=max(0,duracion-2.4)
        filtros.extend([f"[1:v]format=rgba,fade=t=in:st={inicio:.4f}:d=0.45:alpha=1[logo]",
                        "[base][logo]overlay=(W-w)/2:(H-h)/2:shortest=1[spot]"])
        pista="[spot]"
    destino=carpeta/"campana.mp4"
    ejecutar(["-v","error","-y",*entradas,"-filter_complex",";".join(filtros),"-map",pista,
              *([] if silencio else ["-map","0:a:0?","-c:a","aac"]),"-t",str(duracion),
              "-c:v","libx264","-threads","2","-preset","veryfast","-crf","18","-pix_fmt","yuv420p",
              "-movflags","+faststart",str(destino)],tiempo=90)
    return destino,datos.get("id")


def validar_movimiento(ruta):
    raw=ejecutar(["-v","error",*entrada_video(ruta),"-an","-vf","fps=2,scale=48:48,format=gray",
                  "-pix_fmt","gray","-frames:v","32","-f","rawvideo","pipe:1"],salida=True,tiempo=25)
    cuadros=np.frombuffer(raw,np.uint8).reshape(-1,48*48)
    if len(cuadros)<4:
        raise AjusteNoAplicado("La exportación es demasiado corta para verificar el movimiento.")
    cambios=np.abs(np.diff(cuadros.astype("int16"),axis=0)).mean(axis=1)
    if (cambios>1.0).mean()<.5:
        raise AjusteNoAplicado("Detecté muy poco movimiento en el reel exportado.")


def motor_para_pedido(texto, anterior):
    """Una plantilla solo satisface un pedido explícito de animación del logo."""
    orden = normalizar(texto)
    generativo = re.search(r"\b(?:cinematografic\w*|generativ\w*|fantastic\w*|escenas? originales?|desde cero)\b", orden)
    animacion = re.search(r"\b(?:animacion (?:de (?:la |mi )?marca|del? logo)|anima(?:r|me)? (?:el |mi |este )?logo|plantilla (?:de |del )logo)\b", orden)
    if not generativo and (animacion or anterior.get("tipo") == "animacion"):
        return "animacion"
    return "generativo"


def crear_campana(carpeta, texto, cliente, modelo, formato, avisar):
    ficha=carpeta/"campana.json"
    anterior=json.loads(ficha.read_text()) if ficha.is_file() else {}
    logo=carpeta/"logo-campana.png"
    if not logo.is_file():
        guardado=logo_guardado(texto)
        if guardado:
            with Image.open(guardado) as im:
                ImageOps.exif_transpose(im).convert("RGBA").save(logo)
    tiene_logo=logo.is_file()
    if not tiene_logo and re.search(r"\blogo(?:tipo)?\b",normalizar(texto)):
        raise AjusteNoAplicado("Necesito el logo oficial de esa marca. Puedes adjuntarlo junto con el pedido.")
    motor = motor_para_pedido(texto, anterior)
    generativo = motor == "generativo"
    if generativo and not (os.environ.get("GEMINI_API_KEY") and os.environ.get("AURA_VIDEO_GENERATIVO") == "1"):
        raise AjusteNoAplicado("El anuncio requiere generar escenas originales. El generador de video de AURA Web todavía no está activado.")
    if not generativo and (not tiene_logo or re.search(r"\b(?:fotorrealist\w*|accion real|actores?|personajes?|dialog\w*|fantastic\w*|dragon\w*|generativ\w*)\b",normalizar(texto))):
        raise AjusteNoAplicado("Para crear esa escena hace falta activar el generador de video en AURA Web.")
    direccion=dirigir(texto,cliente,modelo,anterior.get("direccion"),logo if tiene_logo else None)
    if not generativo and direccion.get("requiere_generador"):
        raise AjusteNoAplicado("Para representar la acción que pediste, activa el generador de video en AURA Web.")
    silencio=bool(re.search(r"\b(?:sin (?:audio|sonido|musica)|silencio)\b",normalizar(texto))) or (anterior.get("silencio",False) and not re.search(r"\b(?:audio|musica|sonido)\b",normalizar(texto)))
    formato=formato if not anterior or re.search(r"vertical|horizontal|cuadrado|9:16|16:9",texto,re.I) else anterior["formato"]
    if generativo:
        archivo,interaccion=generar_toma(carpeta,direccion,formato,logo if tiene_logo else None,silencio,anterior.get("interaccion"))
    else:
        archivo=renderizar(carpeta,logo,formato,direccion,silencio=silencio,avisar=avisar)
        interaccion=None
    validar_movimiento(archivo)
    if generativo:
        revisar_identidad(archivo, logo if tiene_logo else None, texto, cliente, modelo)
    ficha.write_text(json.dumps({"formato":formato,"direccion":direccion,"tipo":motor,"silencio":silencio,"interaccion":interaccion},ensure_ascii=False))
    return archivo
