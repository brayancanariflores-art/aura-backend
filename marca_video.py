"""Edición localizada de una marca sobre el video fuente."""
import base64
import io
import json
import math
from pathlib import Path
import subprocess
from bisect import bisect_right

from PIL import Image, ImageDraw, ImageOps

from estudio_medios import ejecutar, entrada_video, ffmpeg, inspeccionar


def localizar(clip, instruccion, cliente, modelo, duracion):
    # Muestrear el giro de objetos pequeños sin enviar más de sesenta vistas.
    fps = min(4, 60 / duracion)
    datos = ejecutar(["-v", "error", *entrada_video(clip), "-map", "0:v:0", "-an",
                      "-vf", f"fps={fps:.6f}:round=up,scale=216:312:force_original_aspect_ratio=decrease,pad=216:312:(ow-iw)/2:(oh-ih)/2",
                      "-frames:v", "62", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                     salida=True, tiempo=50)
    tam = 216 * 312 * 3
    total = len(datos) // tam
    if not total:
        raise ValueError("No se pudo leer el video.")
    partes = [{"type": "text", "text": instruccion}]
    for inicio in range(0, total, 8):
        lote = min(8, total - inicio)
        hoja = Image.new("RGB", (4 * 216, math.ceil(lote / 4) * 332))
        for j in range(lote):
            i = inicio + j
            foto = Image.frombytes("RGB", (216, 312), datos[i*tam:(i+1)*tam])
            x, y = j % 4 * 216, j // 4 * 332
            hoja.paste(foto, (x, y))
            ImageDraw.Draw(hoja).text((x + 4, y + 312), f"F{i}", fill="white")
        buf = io.BytesIO()
        hoja.save(buf, format="JPEG", quality=86)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        partes.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    respuesta = cliente.with_options(timeout=85, max_retries=1).chat.completions.create(
        model=modelo, response_format={"type": "json_object"}, messages=[
            {"role": "system", "content":
             "Localiza exclusivamente la marca existente que el usuario quiere sustituir en el objeto y las tomas indicados. "
             "Cada celda contiene un fotograma de 216 por 312 píxeles, incluida su banda negra si hay encuadre, "
             "seguido de la etiqueta F y su número. Las coordenadas son locales a la celda y excluyen la etiqueta. "
             "Para cada fotograma donde aparece esa marca, devuelve [numero,x_izquierda,y_arriba,x_derecha,y_abajo] "
             "en píxeles. Ajusta los bordes al distintivo completo incluida su ilustración. Sigue su perspectiva y "
             "la rotación del objeto. Omite los fotogramas donde no se ve, está tapado o hay otra marca. "
             "Si se pide el vaso del final, omite las marcas del surtidor, las otras tomas y otros vasos. "
             "Devuelve solo JSON con la lista cajas. Si no puedes identificarla, devuelve la lista vacía."},
            {"role": "user", "content": partes}])
    cajas = json.loads(respuesta.choices[0].message.content)["cajas"]
    return cajas, fps, total


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
    cajas,fps,total = localizar(clip,texto,cliente,modelo,info["duracion"])
    grupos = validar_cajas(cajas,fps,total,w,h,info["duracion"])
    archivo = carpeta / "editado.mp4"
    silenciar = any(palabra in texto.lower() for palabra in ("sin audio", "quita el audio", "elimina el audio", "sin sonido"))
    sustitucion(clip, carpeta/"marca.png", archivo, grupos, (w,h), silenciar)
    if not silenciar:
        info_salida = subprocess.run([ffmpeg(), "-hide_banner", *entrada_video(archivo)], capture_output=True, timeout=10)
        if b"Audio:" in info_salida.stderr:
            (carpeta / "pista-conservada").touch()
    return archivo
