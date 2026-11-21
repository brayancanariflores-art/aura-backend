"""Preparación de copias para redes. Solo se abren archivos locales del estudio."""
import io
import math
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps

FOTOS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEOS = {".mp4", ".mov", ".m4v"}
MEDIDAS = {"vertical": (720, 1280), "horizontal": (1280, 720), "cuadrado": (960, 960)}
MAX_PIXELES = 24_000_000


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def ejecutar(argumentos, *, salida=False, tiempo=65):
    args = [ffmpeg(), "-nostdin", "-hide_banner", "-threads", "2",
            "-filter_threads", "1", "-filter_complex_threads", "1", *argumentos]
    return subprocess.run(args, stdout=subprocess.PIPE if salida else subprocess.DEVNULL,
                          stderr=subprocess.PIPE, timeout=tiempo, check=True).stdout


def entrada_video(ruta):
    return ["-protocol_whitelist", "file,pipe", "-f", "mov", "-enable_drefs", "0",
            "-use_absolute_path", "0", "-i", str(ruta)]


def abrir_foto(ruta):
    with Image.open(ruta, formats=("JPEG", "PNG", "WEBP")) as im:
        if im.width * im.height > MAX_PIXELES:
            raise ValueError("La foto supera los 24 megapíxeles.")
        return ImageOps.exif_transpose(im).convert("RGB")


def inspeccionar(ruta):
    if ruta.suffix in FOTOS:
        imagen = abrir_foto(ruta)
        return {"tipo": "foto", "ancho": imagen.width, "alto": imagen.height}, [imagen]
    # Inspeccionar el contenedor sin decodificar fotogramas ni aceptar otras fuentes.
    info = subprocess.run([ffmpeg(), "-nostdin", "-hide_banner", *entrada_video(ruta)],
                          capture_output=True, timeout=12).stderr.decode(errors="replace")
    reloj = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    dimensiones = re.search(r"Video: [^\n]*?\b(\d{2,5})x(\d{2,5})\b", info)
    if not reloj or not dimensiones:
        raise ValueError("No pude leer el video MP4 o MOV.")
    h, m, s = reloj.groups()
    segundos = int(h) * 3600 + int(m) * 60 + float(s)
    w, alto = map(int, dimensiones.groups())
    if not .1 <= segundos <= 600 or w * alto > 4096 * 2304:
        raise ValueError("Usa un video de hasta 10 minutos y resolución 4K.")
    tiempos = [round(segundos * n, 2) for n in (.07, .35, .65, .90)]
    imagenes = []
    for t in tiempos:
        raw = ejecutar(["-v", "error", "-ss", str(t), *entrada_video(ruta),
                        "-map", "0:v:0", "-frames:v", "1", "-vf",
                        "scale=320:320:force_original_aspect_ratio=decrease",
                        "-c:v", "mjpeg", "-f", "image2pipe", "pipe:1"], salida=True, tiempo=15)
        if raw:
            with Image.open(io.BytesIO(raw)) as im:
                imagenes.append(im.convert("RGB"))
    return {"tipo": "video", "duracion": segundos, "muestras_segundos": tiempos,
            "ancho": w, "alto": alto}, imagenes


def miniatura(imagen):
    imagen = imagen.copy()
    imagen.thumbnail((384, 384))
    buf = io.BytesIO()
    imagen.save(buf, "JPEG", quality=72)
    return buf.getvalue()


def rotulo(medidas, texto):
    w, h = medidas
    capa = Image.new("RGBA", medidas)
    pincel = ImageDraw.Draw(capa)
    try:
        fuente = ImageFont.truetype("DejaVuSans-Bold.ttf", max(25, min(w, h) // 20))
    except OSError:
        fuente = ImageFont.load_default(size=max(25, min(w, h) // 20))
    lineas = []
    actual = ""
    for palabra in texto.split():
        posible = (actual + " " + palabra).strip()
        if pincel.textlength(posible, font=fuente) > w * .79 and actual:
            lineas.append(actual)
            actual = palabra
        else:
            actual = posible
    if actual:
        lineas.append(actual)
    leyenda = "\n".join(lineas[:3])
    izq = int(w * .08)
    y = int(h * .73)
    limites = pincel.multiline_textbbox((izq, y), leyenda, font=fuente, spacing=6)
    pincel.rounded_rectangle((izq - 12, y - 9, min(w, limites[2] + 14), limites[3] + 12),
                             radius=8, fill=(13, 17, 30, 206))
    pincel.multiline_text((izq, y), leyenda, font=fuente, fill="white", spacing=6)
    return capa


def editar(ruta, carpeta, info, plan, formato):
    medidas = MEDIDAS[formato]
    ancho, alto = medidas
    titulo = rotulo(medidas, plan["gancho"])
    # Conservar el encuadre original colocando márgenes cuando cambie el formato.
    if info["tipo"] == "foto":
        foto = ImageOps.contain(abrir_foto(ruta), medidas, Image.Resampling.LANCZOS)
        fondo = Image.new("RGB", medidas, (18, 23, 37))
        fondo.paste(foto, ((ancho - foto.width) // 2, (alto - foto.height) // 2))
        fondo.paste(titulo, mask=titulo.getchannel("A"))
        destino = carpeta / "pieza.jpg"
        fondo.save(destino, quality=91, optimize=True)
        return destino, f"Foto {ancho} × {alto} con el encuadre completo y el titular."
    duracion = min(20.0, info["duracion"])
    inicio = float(plan.get("inicio_segundos", 0))
    if not math.isfinite(inicio):
        inicio = 0
    inicio = min(max(0, inicio), max(0, info["duracion"] - duracion))
    destino = carpeta / "pieza.mp4"
    titulo.save(carpeta / "rotulo.png")
    filtro = (f"[0:v:0]fps=24,scale={ancho}:{alto}:force_original_aspect_ratio=decrease,"
              f"pad={ancho}:{alto}:(ow-iw)/2:(oh-ih)/2:color=0x121725,setsar=1[v];"
              "[v][1:v]overlay=0:0:enable='lt(t,4)'[out]")
    ejecutar(["-v", "error", "-y", "-ss", str(inicio), *entrada_video(ruta),
              "-i", str(carpeta / "rotulo.png"), "-filter_complex", filtro, "-map", "[out]",
              "-map", "0:a:0?", "-t", str(duracion), "-c:v", "libx264", "-threads", "2",
              "-preset", "veryfast", "-crf", "23", "-maxrate", "2500k", "-bufsize", "5000k",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-map_metadata", "-1",
              "-movflags", "+faststart", str(destino)], tiempo=100)
    return destino, f"Clip de {duracion:.1f} s desde {inicio:.1f} s, {ancho} × {alto}, con titular inicial."


def reel_narrado(ruta, carpeta, info, plan, formato, frases, narrar, *, escenas=None):
    """Cada subtítulo dura exactamente lo que dura su fragmento de narración PCM."""
    import wave
    medidas = MEDIDAS[formato]
    w, h = medidas
    with wave.open(str(carpeta / "voz.wav"), "wb") as voz:
        voz.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        inicio = 0.
        marcas = []
        for i, frase in enumerate(frases):
            audio = narrar(frase)
            fin = inicio + len(audio) / 48000
            if not audio or len(audio) % 2 or fin > 60:
                raise ValueError("La narración no cabe en un reel de un minuto.")
            voz.writeframes(audio)
            capa = rotulo(medidas, frase)
            ImageDraw.Draw(capa).text((w * .08, h * .08), "Voz generada por IA", fill="white",
                                      stroke_width=2, stroke_fill="#202030", font=ImageFont.load_default(size=20))
            capa.save(carpeta / f"sub-{i}.png")
            marcas.append((inicio, fin))
            inicio = fin
    entradas = []
    if escenas:
        if len(escenas) != len(marcas):
            raise ValueError("Cada frase necesita su escena.")
        for escena in escenas:
            entradas += ["-i", str(escena)]
        filtros = []
        for i, (desde, hasta) in enumerate(marcas):
            frames = math.ceil(hasta * 24) - math.ceil(desde * 24)
            filtros.append(
                f"[{i}:v]scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
                f"crop={w*2}:{h*2},zoompan=z='1+0.06*on/{max(frames, 1)}':"
                f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d={max(frames, 1)}:s={w}x{h}:fps=24,"
                f"setsar=1[escena{i}]")
        pistas = "".join(f"[escena{i}]" for i in range(len(escenas)))
        filtros.append(f"{pistas}concat=n={len(escenas)}:v=1:a=0[v0]")
        audio_idx = len(escenas)
    else:
        if info["tipo"] == "foto":
            foto = carpeta / "base.jpg"
            abrir_foto(ruta).save(foto)
            entradas = ["-loop", "1", "-framerate", "24", "-i", str(foto)]
        else:
            desde = min(max(0, plan["inicio_segundos"]), max(0, info["duracion"] - inicio))
            entradas = ["-stream_loop", "-1", "-ss", str(desde), *entrada_video(ruta)]
        filtros = [f"[0:v:0]fps=24,scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x121725,setsar=1[v0]"]
        audio_idx = 1
    entradas += ["-i", str(carpeta / "voz.wav")]
    for i, (desde, hasta) in enumerate(marcas):
        entradas += ["-i", str(carpeta / f"sub-{i}.png")]
        filtros.append(f"[v{i}][{i+audio_idx+1}:v]overlay=0:0:enable='gte(t,{desde})*lt(t,{hasta})'[v{i+1}]")
    destino = carpeta / "pieza.mp4"
    ejecutar(["-v", "error", "-y", *entradas, "-filter_complex", ";".join(filtros),
              "-map", f"[v{len(marcas)}]", "-map", f"{audio_idx}:a:0", "-t", str(inicio), "-threads", "2",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-maxrate", "2000k", "-bufsize", "4000k",
              "-c:a", "aac", "-b:a", "96k", "-pix_fmt", "yuv420p", "-map_metadata", "-1",
              "-movflags", "+faststart", str(destino)], tiempo=120)
    return destino, f"Reel de {inicio:.1f} s con voz sintética y {len(marcas)} subtítulos sincronizados por frase."
