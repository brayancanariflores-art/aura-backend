"""Ajustes sobre el metraje adjunto, sin generar escenas nuevas."""
import json
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException
from fastapi.responses import Response

from estudio_medios import ejecutar


def plan_video(texto, cliente, modelo):
    simple = texto.casefold().strip(" .¡!¿?")
    if re.fullmatch(r"(?:quita(?:le)? (?:el )?audio|sin audio|silencia(?: el video|lo)?)", simple):
        return {"silencio": True}
    if re.fullmatch(r"(?:(?:pon(?:lo)?|pasa(?:lo)?) (?:en |a )?)?(?:blanco y negro|escala de grises)", simple):
        return {"gris": True}
    if not cliente:
        raise HTTPException(503, "El editor de video no está configurado.")
    respuesta = cliente.chat.completions.create(model=modelo, response_format={"type":"json_object"},
        messages=[{"role":"system","content":(
            "Traduce la orden a un ajuste del video original. Campos opcionales: inicio y fin en segundos "
            "(intervalo que conservar), duracion en segundos desde inicio, silencio (booleano), gris "
            "(blanco y negro, booleano), giro (90, 180 o 270 grados), brillo (-0.5 a 0.5, 0 neutro), "
            "contraste (0.5 a 2, 1 neutro), saturacion (0 a 3, 1 neutro). Si el pedido necesita "
            "operaciones distintas, o falta un dato indispensable, devuelve solo el campo detalle "
            "explicando en español qué falta o qué ajuste se admite. No inventes operaciones ni tiempos.")},
            {"role":"user","content":texto}])
    return json.loads(respuesta.choices[0].message.content)


def editar_video(tipo, datos, texto, cliente, modelo):
    plan = plan_video(texto, cliente, modelo)
    if not isinstance(plan, dict) or plan.get("detalle") or not plan:
        detalle = plan.get("detalle") if isinstance(plan, dict) else None
        raise HTTPException(422, str(detalle or "Indica qué cambio quieres aplicar al video.")[:280])

    def numero(campo, minimo, maximo):
        valor = plan[campo]
        if type(valor) not in (float, int) or not math.isfinite(valor) or not minimo <= valor <= maximo:
            raise HTTPException(422, "El ajuste solicitado está fuera del rango admitido.")
        return valor

    filtros = []
    if plan.get("gris") is True:
        filtros.append("hue=s=0")
    if "giro" in plan:
        giro = numero("giro", 0, 270)
        rotacion = {90:["transpose=1"], 180:["hflip","vflip"], 270:["transpose=2"]}
        if giro not in rotacion:
            raise HTTPException(422, "El giro debe ser de 90, 180 o 270 grados.")
        filtros.extend(rotacion[giro])
    color = []
    for campo, opcion, limites in (("brillo","brightness",(-.5,.5)), ("contraste","contrast",(.5,2)),
                                    ("saturacion","saturation",(0,3))):
        if campo in plan:
            color.append(f"{opcion}={numero(campo, *limites)}")
    if color:
        filtros.append("eq=" + ":".join(color))
    con_audio = plan.get("silencio") is not True
    tiempos = []
    inicio = numero("inicio", 0, 86400) if "inicio" in plan else 0
    if inicio:
        tiempos.extend(["-ss", str(inicio)])
    if "fin" in plan or "duracion" in plan:
        duracion = numero("fin", 0, 86400) - inicio if "fin" in plan else numero("duracion", 0, 86400)
        if duracion <= 0:
            raise HTTPException(422, "El final debe ser posterior al inicio.")
        tiempos.extend(["-t", str(duracion)])
    if not (filtros or tiempos or not con_audio):
        raise HTTPException(422, "Indica un cambio de duración, audio, giro o color.")
    with TemporaryDirectory(prefix="aura-video-") as nombre:
        entrada = Path(nombre) / "entrada"
        salida = Path(nombre) / "editado.mp4"
        entrada.write_bytes(datos)
        demux = ["-f","matroska"] if tipo == "video/webm" else ["-f","mov","-enable_drefs","0","-use_absolute_path","0"]
        ejecutar(["-v","error","-protocol_whitelist","file,pipe",*demux, "-i", str(entrada),
                  *tiempos,"-map","0:v:0",*(["-map","0:a:0?","-c:a","aac"] if con_audio else ["-an"]),
                  "-vf", ",".join([*filtros,"scale=trunc(iw/2)*2:trunc(ih/2)*2"]),
                  "-threads","2","-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
                  "-map_metadata","-1","-movflags","+faststart",str(salida)], tiempo=145)
        if salida.stat().st_size < 1024:
            raise HTTPException(422, "El intervalo solicitado no contiene video.")
        return Response(salida.read_bytes(), media_type="video/mp4")
