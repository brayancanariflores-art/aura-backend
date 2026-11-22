"""Seguimiento de una marca visible mediante tinta, continuidad temporal y revisión visual."""
import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from estudio_medios import ejecutar, entrada_video


def canales(foto):
    return np.asarray(Image.fromarray(foto).convert("HSV"), dtype=np.int16).transpose(2, 0, 1)


def tinta(canales_hsv, tono):
    matiz, saturacion, valor = canales_hsv
    distancia = np.abs((matiz - tono + 128) % 256 - 128)
    return (distancia <= 8) & (saturacion > 55) & (valor > 65)


def anclas(canales_hsv):
    """Proponer regiones cromáticas separables; el modelo identifica el objeto solicitado."""
    encontrados = []
    alto, ancho = canales_hsv.shape[1:]
    for tono in range(0, 256, 16):
        mascara = tinta(canales_hsv, tono)
        regiones, _ = ndimage.label(ndimage.binary_closing(mascara, structure=np.ones((6,7))))
        for etiqueta, ventana in enumerate(ndimage.find_objects(regiones), 1):
            if ventana is None:
                continue
            y, x = ventana
            w, h = x.stop-x.start, y.stop-y.start
            cuantos = (mascara[ventana] & (regiones[ventana] == etiqueta)).sum()
            densidad = cuantos / (w*h)
            if not (.028*ancho <= w <= .65*ancho and .016*alto <= h <= .5*alto
                    and cuantos >= .001*mascara.size and .12 < densidad < .86):
                continue
            if min(x.start, y.start, ancho-x.stop, alto-y.stop) < 4:
                continue
            alrededor = mascara[max(0,y.start-h//3):y.stop+h//3, max(0,x.start-w//3):x.stop+w//3]
            fuera = (alrededor.sum()-mascara[ventana].sum()) / max(1, alrededor.size-w*h)
            puntaje = np.sqrt(cuantos)*densidad / (.20+fuera)
            encontrados.append((float(puntaje), tono, [x.start, y.start, x.stop, y.stop]))
    return sorted(encontrados, reverse=True)[:12]


def tramo(histograma, hueco, umbral):
    histograma = ndimage.uniform_filter1d(histograma.astype(float), 3)
    indices = np.flatnonzero(histograma >= max(1.8, umbral*histograma.max()))
    if not indices.size:
        return indices
    grupos = np.split(indices, np.flatnonzero(np.diff(indices)>hueco)+1)
    return max(grupos, key=lambda grupo: histograma[grupo].sum())


def seguir(frames, origen, tono, inicial):
    cajas = {origen: np.array(inicial, dtype=float)}
    for direccion in (1, -1):
        previa = cajas[origen]
        velocidad = np.zeros(4)
        ancho_base, alto_base = previa[2:] - previa[:2]
        limite = len(frames) if direccion == 1 else -1
        for i in range(origen+direccion, limite, direccion):
            foto = frames[i]
            anterior = frames[i-direccion, ::8, ::8].astype(np.int16)
            if np.abs(anterior-foto[::8,::8]).mean() > 45:
                break
            predicha = previa + velocidad*.6
            w, h = np.maximum(predicha[2:]-predicha[:2], [2,3])
            margen = np.array([max(6,w*.25,ancho_base*.16), max(5,h*.16,alto_base*.10)])
            x, y = np.maximum(0, np.floor(predicha[:2]-margen)).astype(int)
            r, b = np.minimum([foto.shape[1],foto.shape[0]], np.ceil(predicha[2:]+margen)).astype(int)
            if r<=x or b<=y:
                break
            mascara = tinta(canales(foto[y:b,x:r]), tono)
            if mascara.sum() < max(8,.045*w*h):
                break
            hueco_x = max(3,w*.17,ancho_base*.18)
            columnas = tramo(mascara.sum(axis=0), hueco_x, .17)
            if not len(columnas):
                break
            filas = tramo(mascara[:,columnas[0]:columnas[-1]+1].sum(axis=1),
                         max(4,h*.18,alto_base*.15), .145)
            if not len(filas):
                break
            columnas = tramo(mascara[filas[0]:filas[-1]+1].sum(axis=0), hueco_x, .15)
            if not len(columnas):
                break
            actual = np.array([x+columnas[0], y+filas[0], x+columnas[-1]+1, y+filas[-1]+1], float)
            cw, ch = actual[2:]-actual[:2]
            if ch < max(4,h*.42) or cw<2 or cw*ch < ancho_base*alto_base*.008:
                break
            velocidad = .5*velocidad + .5*(actual-previa)
            cajas[i] = previa = actual
    return cajas


def imagen_chat(imagen):
    buf = io.BytesIO()
    imagen.save(buf, format="JPEG", quality=90)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type":"image_url", "image_url":{"url":uri, "detail":"high"}}


def escoger_marca(servicio, modelo, texto, foto, candidatos):
    original = Image.fromarray(foto)
    partes = [{"type":"text", "text":texto}, imagen_chat(original)]
    for indice, (_, _, caja) in enumerate(candidatos,1):
        mini = original.copy()
        ImageDraw.Draw(mini).rectangle(caja, outline="cyan", width=2)
        x,y,r,b = caja
        margen = max(4, round(min(r-x,b-y)*.28))
        detalle = original.crop((max(0,x-margen),max(0,y-margen),
                                 min(original.width,r+margen),min(original.height,b+margen)))
        detalle.thumbnail((160,160))
        mini.thumbnail((165,210))
        vista = Image.new("RGB",(mini.width+detalle.width,max(mini.height,detalle.height)))
        vista.paste(mini); vista.paste(detalle,(mini.width,(vista.height-detalle.height)//2))
        partes.extend([{"type":"text","text":f"Candidato {indice}"},imagen_chat(vista)])
    guia = ("El primer fotograma es el original. Escoge la caja que encierra la tinta de la marca "
            "impresa solicitada en el objeto, incluso si omite pequeñas partes negras o blancas del dibujo. "
            "El cian solo indica una propuesta. Descarta reflejos, superficies enteras, el líquido y "
            "otras etiquetas. Si no puedes identificarla inequívocamente, devuelve 0. Indica si hay "
            "elementos del mismo dibujo fuera de la caja a su izquierda o derecha o por ambos lados. "
            'Responde JSON con un entero \"candidato\" y \"extension\": \"izquierda\", \"derecha\" o \"ambos\".')
    respuesta=servicio.chat.completions.create(model=modelo,response_format={"type":"json_object"},
        messages=[{"role":"system","content":guia},{"role":"user","content":partes}])
    seleccion=json.loads(respuesta.choices[0].message.content or "{}")
    indice=seleccion.get("candidato")
    if type(indice) is not int or not 1<=indice<=len(candidatos):
        return None
    _,tono,caja=candidatos[indice-1]
    return tono,caja,seleccion.get("extension","ambos")


def recuperar_letras(frames,tono,pista):
    """Reunir letras separadas que pudieron quedar fuera de la ventana del fotograma anterior."""
    ids=sorted(pista)
    arr=np.stack([pista[i] for i in ids])
    w0,h0=np.percentile(arr[:,2:]-arr[:,:2],90,axis=0)
    resultado={}
    for j,i in enumerate(ids):
        izq,y,der,b=arr[j]
        w,h=der-izq,b-y
        # Una palabra puede tener varios grupos de tinta aunque comparta una sola etiqueta.
        x1,y1=np.maximum(0,np.floor([izq-.65*w0,y-.08*h0])).astype(int)
        x2,y2=np.minimum([frames.shape[2],frames.shape[1]],np.ceil([der+.65*w0,b+.08*h0])).astype(int)
        mask=tinta(canales(frames[i,y1:y2,x1:x2]),tono)
        conteo=mask.sum(axis=0)
        suave=ndimage.uniform_filter1d(conteo.astype(float),3)
        cols=np.flatnonzero(suave>=max(2.6,.20*suave.max()))
        if cols.size:
            hueco=max(5, min(w0*.255,max(w*.38,w0*.18)))
            grupos=np.split(cols,np.flatnonzero(np.diff(cols)>hueco)+1)
            def puntuacion(c):
                comun=max(0,min(c[-1]+1,der-x1)-max(c[0],izq-x1))
                return suave[c].sum()*min(1,comun/max(2,.30*w))
            cs=max(grupos,key=puntuacion)
            if puntuacion(cs)>0:
                izq,der=x1+cs[0], x1+cs[-1]+1
        resultado[i]=np.array([izq,y,der,b],float)
    return resultado


def trayectoria_adaptada(items, lado, fps, tam, formato, duracion):
    indices=sorted(items)
    cajas=np.array([items[i] for i in indices])
    cajas=ndimage.median_filter(cajas,size=(3,1),mode="nearest")
    # Suavizar los cambios de luz sin borrar el momento en que entra el dibujo.
    cajas=ndimage.gaussian_filter1d(cajas, .8, axis=0, mode="nearest")
    anchos=cajas[:,2]-cajas[:,0]
    ancho_frontal=max(1,np.percentile(anchos,85))
    alto,ancho=formato
    escala_x,escala_y=tam[0]/ancho,tam[1]/alto
    limite=max(1,min(len(cajas)-1,int(np.argmax(anchos >= .7*ancho_frontal))))
    giro=None
    if anchos[0] < ancho_frontal*.55 and limite>0:
        giro="izquierda" if abs(cajas[limite,0]-cajas[0,0])>abs(cajas[limite,2]-cajas[0,2]) else "derecha"
    camino=[]
    for i,(x,y,r,b) in zip(indices,cajas):
        w,h=r-x,b-y
        ratio=min(1,w/ancho_frontal)
        apertura=min(1,max(0,(ratio-.38)/.42))
        frontal=max(w*.18,min(h*.21,ancho_frontal*.18))
        menor=w*(.025+.15*apertura)
        extra=min(w*(1.26-.30*apertura),max(frontal,ancho_frontal*(.44-.21*apertura)))
        direccion=giro if giro and ratio<.78 else lado
        izq,der=(menor,extra) if direccion=="derecha" else (extra,menor)
        if direccion=="ambos" and ratio>=.78:
            izq,der=frontal*1.13,frontal
        x1,x2=max(0,x-izq),min(ancho,r+der)
        # El trazo blanco o negro que rodea la tinta conserva la altura al verse de perfil.
        y1=max(0,y-h*(.115-.035*apertura))
        y2=min(alto,b+h*(.225-.105*apertura))
        camino.append([i/fps,x1*escala_x,y1*escala_y,(x2-x1)*escala_x,(y2-y1)*escala_y])
    if giro and indices[0]>0 and anchos[0]<.35*ancho_frontal:
        tiempo,x,y,w,h=camino[0]
        borde=x+w if giro=="izquierda" else x
        camino.insert(0,[tiempo-1/fps,min(tam[0]-1,borde),y+h*.31,1,max(1,h*.38)])
    return [(camino,min(duracion,(indices[-1]+1)/fps))]


def comprobar_objeto(servicio, modelo, texto, frames, cajas):
    """Nunca entregar una región de color sin comprobar qué objeto contiene durante el giro."""
    indices=sorted(cajas)
    puntos=sorted(set(indices[int(p*(len(indices)-1))] for p in (.06,.18,.45,.74,.96)))
    partes=[{"type":"text","text":texto}]
    for i in puntos:
        im=Image.fromarray(frames[i])
        ImageDraw.Draw(im).rectangle(tuple(cajas[i]),outline="cyan",width=2)
        partes.append(imagen_chat(im))
    respuesta=servicio.chat.completions.create(model=modelo,response_format={"type":"json_object"},
        messages=[{"role":"system","content":'Las cajas cian delimitan la tinta principal de una marca en varios momentos. '
                   '¿Siguen la misma pegatina o texto impreso en el objeto que pidió el usuario, incluso de perfil? '
                   'Devuelve JSON {"misma_marca":true} si sí. Si señalan objetos diferentes o solo el fondo, devuelve false.'},
                  {"role":"user","content":partes}])
    return json.loads(respuesta.choices[0].message.content or "{}").get("misma_marca") is True


def buscar_al_final(clip, texto, cliente, modelo, duracion, tam):
    """Reconoce la marca en una vista nítida y mide su posición hacia atrás y hacia adelante."""
    fps=15
    escala=min(384/max(tam),1)
    w,h=(max(4,2*round(v*escala/2)) for v in tam)
    servicio=cliente.with_options(timeout=45,max_retries=1)
    with tempfile.TemporaryDirectory(dir=clip.parent) as temporal:
        ruta=Path(temporal)/"muestras.rgb"
        ejecutar(["-y","-v","error",*entrada_video(clip),"-map","0:v:0","-an","-vf",
                  f"fps={fps},scale={w}:{h}","-frames:v",str(int(duracion*fps)+2),
                  "-pix_fmt","rgb24","-f","rawvideo",str(ruta)],tiempo=55)
        if ruta.stat().st_size<4*w*h*3:
            return None
        frames=np.memmap(ruta,dtype="uint8",mode="r").reshape(-1,h,w,3)
        for pos in (.99,.81,.68):
            i=round((len(frames)-1)*pos)
            opciones=anclas(canales(frames[i]))
            if not opciones: continue
            hallada=escoger_marca(servicio,modelo,texto,frames[i],opciones)
            if hallada is None: continue
            tono,caja,lado=hallada
            pista=seguir(frames,i,tono,caja)
            pista=recuperar_letras(frames,tono,pista)
            if len(pista)>=max(4,fps//2) and comprobar_objeto(servicio,modelo,texto,frames,pista):
                return trayectoria_adaptada(pista,lado,fps,tam,(h,w),duracion)
    return None
