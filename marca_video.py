"""Edición localizada de una marca sobre el video fuente."""
import base64
import io
import json
import math
import logging
import re
from pathlib import Path
import subprocess
from bisect import bisect_right

from PIL import Image, ImageDraw, ImageOps

from estudio_medios import ejecutar, entrada_video, ffmpeg, inspeccionar


# Se solicita un objeto con campos nombrados para cada vista. Los ejes son
# relativos al cuadro completo de video y no dependen de mosaicos ni relleno.
CAMPOS_CAJA = ("fotograma", "izquierda", "arriba", "derecha", "abajo")
FORMATO_TRAYECTORIA = {
    "type": "json_schema",
    "json_schema": {
        "name": "localizacion_visual",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"cajas": {"type": "array", "items": {
                "type": "object",
                "properties": {k: {"type": "integer"} for k in CAMPOS_CAJA},
                "required": list(CAMPOS_CAJA), "additionalProperties": False,
            }}},
            "required": ["cajas"], "additionalProperties": False,
        },
    },
}
log = logging.getLogger(__name__)


def localizar(clip, instruccion, cliente, modelo, duracion):
    fps = min(4, 60 / duracion)
    # Si la orden trae un inicio explícito, las imágenes de otras tomas no
    # deben influir en el reconocimiento de la marca.
    inicio = re.search(r"(?:desde|a partir de)\s+(?:el\s+)?(?:segundo\s+)?(\d+(?:[.,]\d+)?)\s*(?:s(?:egundos?)?\b|hasta)", instruccion, re.I)
    desde = float(inicio[1].replace(",", ".")) if inicio else 0.0
    if not 0 <= desde < duracion:
        raise ValueError("El inicio está fuera del video.")
    datos = ejecutar(["-v", "error", *entrada_video(clip), "-map", "0:v:0", "-an",
                      "-vf", f"fps={fps:.6f}:round=up,scale=216:312:force_original_aspect_ratio=decrease,pad=216:312:(ow-iw)/2:(oh-ih)/2",
                      "-frames:v", "62", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                     salida=True, tiempo=50)
    tam = 216 * 312 * 3
    total = len(datos) // tam
    if not total:
        raise ValueError("No se pudo leer el video.")
    cajas = []
    servicio = cliente.with_options(timeout=65, max_retries=1)
    for primer in range(math.ceil(desde * fps), total, 12):
        partes = [{"type": "text", "text": instruccion}]
        ultimo = min(primer + 12, total)
        for i in range(primer, ultimo):
            foto = Image.frombytes("RGB", (216, 312), datos[i*tam:(i+1)*tam])
            buf = io.BytesIO()
            foto.save(buf, format="JPEG", quality=90)
            url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            partes.extend([{"type": "text", "text": f"Fotograma {i}, tiempo {i/fps:.2f} segundos"},
                           {"type": "image_url", "image_url": {"url": url, "detail": "high"}}])
        respuesta = servicio.chat.completions.create(
            model=modelo, response_format=FORMATO_TRAYECTORIA, messages=[
                {"role": "system", "content":
                 "Eres un localizador de objetos. Cada imagen es un fotograma completo, precedido de su número. "
                 "Encuentra la marca que ya aparece impresa en el objeto descrito en la orden. Usa el número de "
                 "fotograma indicado. Devuelve la caja ajustada al dibujo completo visible. Coordenadas enteras "
                 "de 0 a 1000 en ambos ejes: izquierda y derecha como fracción del ancho, arriba y abajo como "
                 "fracción del alto. Incluye las bandas negras en el sistema de coordenadas. Un registro por "
                 "fotograma visible. Omite los que no muestran la marca en ese objeto. Solo JSON."},
                {"role": "user", "content": partes}])
        contenido = respuesta.choices[0].message.content
        if not contenido:
            raise ValueError("Análisis visual vacío.")
        valores = json.loads(contenido).get("cajas")
        if not isinstance(valores, list):
            raise ValueError("Respuesta de localización ilegible.")
        for fila in valores:
            if not isinstance(fila, dict) or not all(type(fila.get(k)) is int for k in CAMPOS_CAJA):
                continue
            i, x1, y1, x2, y2 = (fila[k] for k in CAMPOS_CAJA)
            if primer <= i < ultimo and 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
                cajas.append([i, x1*.216, y1*.312, x2*.216, y2*.312])
    unicas = {fila[0]: fila for fila in cajas}
    log.info("Localización visual: %d fotogramas con marca de %d", len(unicas), total)
    return [unicas[i] for i in sorted(unicas)], fps, total


def trayectoria_guiada(texto, ancho, alto, duracion):
    """Puntos de edición opcionales aportados en la orden, con posiciones porcentuales."""
    marcador = "guía de movimiento:"
    if marcador not in texto.lower():
        return None
    linea = texto.lower().rsplit(marcador, 1)[1].partition("\n")[0].strip()
    try:
        filas = json.loads(linea)
    except (ValueError, RecursionError):
        raise ValueError("La guía de movimiento no es válida.") from None
    if not isinstance(filas, list) or not 2 <= len(filas) <= 160:
        raise ValueError("La guía debe indicar al menos el inicio y el fin.")
    grupo = []
    for fila in filas:
        if not isinstance(fila, list) or len(fila) != 5 or not all(
            type(v) in (float, int) and math.isfinite(v) for v in fila
        ):
            raise ValueError("La guía contiene un punto inválido.")
        t, x1, y1, x2, y2 = fila
        if not (0 <= t <= duracion + .05 and 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100):
            raise ValueError("La guía está fuera de los límites del video.")
        t = min(t, duracion)
        if grupo and t <= grupo[-1][0]:
            raise ValueError("Ordena los puntos de la guía por tiempo.")
        grupo.append([t, x1*ancho/100, y1*alto/100, (x2-x1)*ancho/100, (y2-y1)*alto/100])
    return [(grupo, grupo[-1][0])]


def validar_cajas(cajas, fps, total, ancho, alto, duracion):
    escala = min(216/ancho, 312/alto)
    dx, dy = (216-ancho*escala)/2, (312-alto*escala)/2
    grupos, anterior = [], -2
    if not isinstance(cajas, list) or len(cajas) > total:
        raise ValueError("Posición de la marca inválida.")
    for fila in cajas:
        i, x1, y1, x2, y2 = fila
        if type(i) is not int or not anterior < i < total or not all(
            type(v) in (int, float) and math.isfinite(v) for v in (x1,y1,x2,y2)
        ):
            raise ValueError("Trayectoria inválida.")
        x1, x2 = max(dx, x1), min(216-dx, x2)
        y1, y2 = max(dy, y1), min(312-dy, y2)
        if x2-x1 < 2 or y2-y1 < 2 or (x2-x1)*(y2-y1) > ancho*alto*escala**2*.70:
            raise ValueError("No se pudo delimitar la marca.")
        if i != anterior + 1:
            grupos.append([])
        grupos[-1].append([i/fps, (x1-dx)/escala, (y1-dy)/escala,
                            (x2-x1)/escala, (y2-y1)/escala])
        anterior = i
    if not grupos:
        raise ValueError("No pude localizar la marca en el video.")
    return [(g, min(duracion, g[-1][0] + 1/fps)) for g in grupos]


def sustitucion(clip, logo, archivo, grupos, tam, silenciar=False):
    # Una capa transparente en coordenadas del original. El reloj de la pista de video
    # original determina los fotogramas de salida, incluso para clips de tasa variable.
    ancho, alto = tam
    izquierdas = [p[1] for grupo, _ in grupos for p in grupo]
    superiores = [p[2] for grupo, _ in grupos for p in grupo]
    origen_x = max(0, math.floor(min(izquierdas))-2)
    origen_y = max(0, math.floor(min(superiores))-2)
    der = min(ancho, math.ceil(max(p[1]+p[3] for grupo, _ in grupos for p in grupo))+2)
    inf = min(alto, math.ceil(max(p[2]+p[4] for grupo, _ in grupos for p in grupo))+2)
    # Mantener dimensiones pares para el muestreo de crominancia.
    origen_x -= origen_x % 2
    origen_y -= origen_y % 2
    size = (der-origen_x, inf-origen_y)
    with Image.open(logo) as im:
        arte = ImageOps.exif_transpose(im).convert("RGBA")
        arte.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    # Conservar el arte completo, incluso si llega como JPG de fondo opaco.
    # Calcular el margen que requiere su contorno para caber en una placa ovalada.
    fondo = arte.getpixel((0, 0))
    fondo = fondo[:3] if fondo[3] > 240 else (250, 250, 250)
    contorno = arte.copy()
    contorno.thumbnail((128,128))
    radio = 1.0
    for y in range(contorno.height):
        for x in range(contorno.width):
            r,g,b,a = contorno.getpixel((x,y))
            if a > 32 and max(abs(c-d) for c,d in zip((r,g,b),fondo)) > 22:
                radio = max(radio, math.hypot((2*x+1)/contorno.width-1,(2*y+1)/contorno.height-1))
    placa = Image.new("RGBA", (math.ceil(arte.width*(radio+.02)),math.ceil(arte.height*(radio+.02))), (*fondo,255))
    placa.alpha_composite(arte, ((placa.width-arte.width)//2,(placa.height-arte.height)//2))
    mask = Image.new("L", placa.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0,0,placa.width-1,placa.height-1), radius=min(placa.size)*0.42, fill=255)
    placa.putalpha(mask)
    arte = placa
    tiempos = [[p[0] for p in g] for g, _ in grupos]
    fps = 30
    inicio = math.floor(grupos[0][0][0][0]*fps)/fps
    final = grupos[-1][1]
    capa = Path(archivo).with_suffix(".overlay.mov")
    args = [ffmpeg(), "-y", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0", "-threads", "1",
            "-c:v", "qtrle", str(capa)]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    vacio = Image.new("RGBA", size)
    try:
        for frame in range(math.ceil((final-inicio)*fps)):
            t = inicio + frame/fps
            lienzo = vacio.copy()
            for (g, fin), ts in zip(grupos, tiempos):
                if not ts[0]-1e-7 <= t < fin:
                    continue
                i = max(0, bisect_right(ts,t)-1)
                p, q = g[i], g[min(i+1,len(g)-1)]
                f = max(0,min(1,(t-p[0])/(q[0]-p[0]))) if q[0] > p[0] else 0
                x,y,w,h = [round(a+f*(b-a)) for a,b in zip(p[1:],q[1:])]
                sello = arte.resize((max(1,w),max(1,h)),Image.Resampling.LANCZOS)
                lienzo.paste(sello,(x-origen_x,y-origen_y))
                break
            proc.stdin.write(lienzo.tobytes())
        proc.stdin.close()
        if proc.wait(timeout=40):
            raise ValueError("Error creando la capa de la marca.")
        ejecutar(["-y","-v","error",*entrada_video(clip),"-itsoffset",f"{inicio:.8f}","-i",str(capa),
                  "-filter_complex",f"[0:v:0][1:v:0]overlay={origen_x}:{origen_y}:eof_action=pass:repeatlast=0:enable='between(t,{inicio:.8f},{final:.8f})'[v]",
                  "-map","[v]",*([] if silenciar else ["-map","0:a:0?"]),"-c:v","libx264","-threads","2","-crf","17","-preset","veryfast",
                  "-pix_fmt","yuv420p","-fps_mode","passthrough","-c:a","copy","-movflags","+faststart",str(archivo)],tiempo=160)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        capa.unlink(missing_ok=True)


def editar_marca(carpeta, texto, cliente, modelo):
    clip = carpeta / "entrada.mp4"
    info, _ = inspeccionar(clip)
    w, h = info["ancho"], info["alto"]
    if info["duracion"] > 60.06 or w*h > 1920*1080:
        raise ValueError("Usa un video de hasta un minuto y resolución Full HD.")
    # El tamaño de visualización incluye la rotación de cámara del teléfono.
    prueba = ejecutar(["-v","error",*entrada_video(clip),"-map","0:v:0","-frames:v","1",
                      "-c:v","mjpeg","-f","image2pipe","pipe:1"],salida=True,tiempo=20)
    with Image.open(io.BytesIO(prueba)) as im:
        w,h = im.size
    grupos = trayectoria_guiada(texto, w, h, info["duracion"])
    if grupos is None:
        cajas, fps, total = localizar(clip, texto, cliente, modelo, info["duracion"])
        grupos = validar_cajas(cajas, fps, total, w, h, info["duracion"])
    archivo = carpeta / "editado.mp4"
    silenciar = any(palabra in texto.lower() for palabra in ("sin audio", "quita el audio", "elimina el audio", "sin sonido"))
    sustitucion(clip, carpeta/"marca.png", archivo, grupos, (w,h), silenciar)
    if not silenciar:
        info_salida = subprocess.run([ffmpeg(), "-hide_banner", *entrada_video(archivo)], capture_output=True, timeout=10)
        if b"Audio:" in info_salida.stderr:
            (carpeta / "pista-conservada").touch()
    return archivo
