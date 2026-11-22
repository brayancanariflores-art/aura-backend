"""Edición localizada de una marca sobre el video fuente."""
import base64
from contextlib import suppress
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
    servicio = cliente.with_options(timeout=65, max_retries=1)
    comienzo = math.ceil(desde * fps)
    propuestas = consultar_vistas(servicio, modelo, instruccion, datos, total, fps, comienzo)
    revisadas = consultar_vistas(servicio, modelo, instruccion, datos, total, fps, comienzo, propuestas)
    validar_acuerdo_visual(propuestas, revisadas)
    log.info("Localización visual: %d propuestas, %d confirmadas", len(propuestas), len(revisadas))
    return [[i, x1*.216, y1*.312, x2*.216, y2*.312]
            for i, (x1, y1, x2, y2) in sorted(revisadas.items())], fps, total


def validar_acuerdo_visual(propuestas, revisadas):
    """Descartar ediciones si las dos lecturas discrepan sobre dónde o cuándo cubrir."""
    todos = set(propuestas) | set(revisadas)
    coincidencias = 0
    for i in propuestas.keys() & revisadas.keys():
        a, b = propuestas[i], revisadas[i]
        ancho = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        alto = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ancho * alto
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        if union and inter / union >= 0.55:
            coincidencias += 1
    if not todos or coincidencias < 0.75 * len(todos):
        raise ValueError("La posición de la marca requiere una guía de movimiento.")


def vista_de_revision(imagen, caja):
    """Ejes en milésimas del cuadro, idénticos a las coordenadas del compositor."""
    ancho, alto = 432, 624
    margen_x, margen_y = 32, 22
    panel = Image.new("RGB", (ancho + margen_x + 16, alto + margen_y + 14), "#191919")
    panel.paste(imagen.resize((ancho, alto), Image.Resampling.LANCZOS), (margen_x, margen_y))
    pincel = ImageDraw.Draw(panel, "RGBA")
    for marca in range(0, 1001, 100):
        x = margen_x + round(marca * ancho / 1000)
        y = margen_y + round(marca * alto / 1000)
        pincel.line((x, margen_y, x, margen_y+alto), fill=(255,255,255,92))
        pincel.line((margen_x, y, margen_x+ancho, y), fill=(255,255,255,92))
        pincel.text((x-9, 3), str(marca), fill="white")
        pincel.text((1, y-5), str(marca), fill="white")
    if caja is not None:
        x1, y1, x2, y2 = caja
        pincel.rectangle((margen_x+x1*ancho/1000, margen_y+y1*alto/1000,
                          margen_x+x2*ancho/1000, margen_y+y2*alto/1000),
                         outline=(0,230,244,255), width=2)
    return panel


def consultar_vistas(servicio, modelo, instruccion, datos, total, fps, comienzo, propuestas=None):
    tam = 216 * 312 * 3
    fin_lote = 12
    indicaciones = (
        "Localiza la marca ORIGINAL impresa en el objeto que describe el pedido. "
        "Cada imagen está precedida de su número de fotograma y tiempo. Devuelve el número original. "
        "Ajusta la caja al borde del diseño visible, incluso si solo se ve de perfil. "
        "No incluyas el cuerpo del objeto, otras marcas ni el espacio alrededor. Omite los fotogramas "
        "en los que esa marca aún no es visible. Usa coordenadas de 0 a 1000 (x de izquierda a derecha, "
        "y de arriba a abajo), con origen en la esquina superior izquierda de la imagen del video."
    )
    if propuestas is not None:
        indicaciones += (
            " Actúas como segundo revisor. El rectángulo cian es una hipótesis falible de otra pasada. "
            "Mira dónde están las letras o el dibujo real y lee los números de los dos ejes. Corrige los "
            "cuatro bordes si el rectángulo está por debajo, por encima, fuera o tiene otro tamaño. "
            "Borra las detecciones de otras tomas. Añade las que falten, incluidas las últimas vistas "
            "si la marca sigue a la vista. Las líneas de la cuadrícula marcan 100 unidades. "
            "Las etiquetas de los ejes están fuera del video y no cuentan en sus coordenadas."
        )
    cajas = {}
    for primer in range(comienzo, total, fin_lote):
        partes = [{"type": "text", "text": instruccion}]
        ultimo = min(primer + fin_lote, total)
        for i in range(primer, ultimo):
            foto = Image.frombytes("RGB", (216, 312), datos[i*tam:(i+1)*tam])
            if propuestas is not None:
                foto = vista_de_revision(foto, propuestas.get(i))
            buf = io.BytesIO()
            foto.save(buf, format="JPEG", quality=90)
            url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            partes.extend([{"type": "text", "text": f"Fotograma {i}, tiempo {i/fps:.2f} segundos"},
                           {"type": "image_url", "image_url": {"url": url, "detail": "high"}}])
        respuesta = servicio.chat.completions.create(
            model=modelo, response_format=FORMATO_TRAYECTORIA, messages=[
                {"role": "system", "content": indicaciones},
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
                cajas[i] = (x1, y1, x2, y2)
    return cajas


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


def preparar_placa(logo):
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
    return placa


def sustitucion(clip, logo, archivo, grupos, tam, silenciar=False, *, impreso=False, curva=False):
    # Una capa transparente en coordenadas del original. El reloj de la pista de video
    # original determina los fotogramas de salida, incluso para clips de tasa variable.
    ancho, alto = tam
    izquierdas = [p[1] for grupo, _ in grupos for p in grupo]
    superiores = [p[2] for grupo, _ in grupos for p in grupo]
    margen = 2
    if impreso:
        margen += math.ceil(max(max(p[3:]) for g, _ in grupos for p in g)*.34)
    origen_x = max(0, math.floor(min(izquierdas))-margen)
    origen_y = max(0, math.floor(min(superiores))-margen)
    der = min(ancho, math.ceil(max(p[1]+p[3] for grupo, _ in grupos for p in grupo))+margen)
    inf = min(alto, math.ceil(max(p[2]+p[4] for grupo, _ in grupos for p in grupo))+margen)
    # Mantener dimensiones pares para el muestreo de crominancia.
    origen_x -= origen_x % 2
    origen_y -= origen_y % 2
    size = (der-origen_x, inf-origen_y)
    if impreso:
        from tinta_video import imprimir, preparar_tinta, vista_frontal
        arte = preparar_tinta(logo)
        vistas = [vista_frontal(g) for g, _ in grupos]
    else:
        arte = preparar_placa(logo)
    tiempos = [[p[0] for p in g] for g, _ in grupos]
    fps = 30
    inicio = math.floor(grupos[0][0][0][0]*fps)/fps
    final = grupos[-1][1]
    capa = Path(archivo).with_suffix(".overlay.mov")
    args = [ffmpeg(), "-y", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0", "-threads", "1",
            "-c:v", "qtrle", str(capa)]
    lector = None
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    vacio = Image.new("RGBA", size)
    try:
        fotogramas = math.ceil((final-inicio)*fps-1e-7)
        if impreso:
            primer = round(inicio*fps)
            filtro = (f"fps={fps},crop={size[0]}:{size[1]}:{origen_x}:{origen_y}:exact=1,"
                      f"tpad=stop_mode=clone:stop_duration=0.1,trim=start_frame={primer}:end_frame={primer+fotogramas},setpts=PTS-STARTPTS")
            lector = subprocess.Popen([ffmpeg(), "-nostdin","-v","error","-threads","2","-filter_threads","1",
                *entrada_video(clip),"-map","0:v:0","-an","-vf",filtro,"-threads","1","-fps_mode","passthrough","-pix_fmt","rgb24",
                "-f","rawvideo","pipe:1"], stdout=subprocess.PIPE)
        for frame in range(fotogramas):
            t = inicio + frame/fps
            lienzo = vacio.copy()
            if lector is not None:
                datos = lector.stdout.read(size[0]*size[1]*3)
                if len(datos) != size[0]*size[1]*3:
                    raise ValueError("No se pudo leer la superficie del video.")
                superficie = Image.frombytes("RGB", size, datos)
            for indice, ((g, fin), ts) in enumerate(zip(grupos, tiempos)):
                if not ts[0]-1e-7 <= t < fin:
                    continue
                i = max(0, bisect_right(ts,t)-1)
                p, q = g[i], g[min(i+1,len(g)-1)]
                f = max(0,min(1,(t-p[0])/(q[0]-p[0]))) if q[0] > p[0] else 0
                x,y,w,h = [round(a+f*(b-a)) for a,b in zip(p[1:],q[1:])]
                caja = (x-origen_x, y-origen_y, max(1,w), max(1,h))
                if impreso:
                    sello, posicion = imprimir(superficie, caja, arte, vistas[indice], curva)
                else:
                    sello = arte.resize(caja[2:], Image.Resampling.LANCZOS)
                    posicion = caja[:2]
                lienzo.paste(sello, posicion)
                break
            proc.stdin.write(lienzo.tobytes())
        proc.stdin.close()
        if lector is not None and lector.wait(timeout=25):
            raise ValueError("Error al leer los fotogramas del video.")
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
        if lector is not None:
            lector.stdout.close()
            if lector.poll() is None:
                lector.kill()
                lector.wait()
        with suppress(BrokenPipeError):
            proc.stdin.close()
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
    if grupos is None and re.search(r"\b(?:al|el|hacia el|sobre el)\s+final\b", texto, re.I) and not re.search(r"\b(?:desde|a partir de)\b",texto,re.I):
        from seguimiento_marca import buscar_al_final
        grupos = buscar_al_final(clip, texto, cliente, modelo, info["duracion"], (w,h))
    if grupos is None:
        cajas, fps, total = localizar(clip, texto, cliente, modelo, info["duracion"])
        grupos = validar_cajas(cajas, fps, total, w, h, info["duracion"])
    archivo = carpeta / "editado.mp4"
    silenciar = any(palabra in texto.lower() for palabra in ("sin audio", "quita el audio", "elimina el audio", "sin sonido"))
    from tinta_video import acabado_impreso
    curva = bool(re.search(r"\b(?:vaso|copa|botella|taza|lata|cilindro)\b", texto, re.I))
    sustitucion(clip, carpeta/"marca.png", archivo, grupos, (w,h), silenciar,
                impreso=acabado_impreso(texto), curva=curva)
    if not silenciar:
        info_salida = subprocess.run([ffmpeg(), "-hide_banner", *entrada_video(archivo)], capture_output=True, timeout=10)
        if b"Audio:" in info_salida.stderr:
            (carpeta / "pista-conservada").touch()
    return archivo
