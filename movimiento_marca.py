"""Animación continua de identidad para campañas con un arte de marca aprobado."""
import hashlib
import math
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage

from estudio_medios import ffmpeg


def recortar_logo(ruta):
    with Image.open(ruta) as origen:
        arte = ImageOps.exif_transpose(origen).convert("RGBA")
        arte.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    pix = np.array(arte)
    # Separar únicamente el papel blanco conectado a la orilla. El blanco de las letras queda intacto.
    claros = pix[:, :, :3].min(2) >= 240
    borde = np.zeros_like(claros)
    borde[[0, -1], :] = claros[[0, -1], :]
    borde[:, [0, -1]] = claros[:, [0, -1]]
    if pix[:, :, 3].min() == 255 and borde.sum() > (arte.width + arte.height):
        papel = ndimage.binary_propagation(borde, mask=claros)
        pix[:, :, 3][papel] = 0
    arte = Image.fromarray(pix)
    return arte.crop(arte.getchannel("A").getbbox() or (0, 0, arte.width, arte.height))


def paisaje(semilla):
    azar = np.random.default_rng(semilla)
    ruido = np.zeros((512, 512), dtype=np.float32)
    for radio, peso in ((28, 1.), (13, .50), (6, .24), (2.5, .1), (.8, .04)):
        capa = ndimage.gaussian_filter(azar.standard_normal(ruido.shape).astype("f"), radio, mode="wrap")
        ruido += peso * capa / max(.001, capa.std())
    y, x = np.indices(ruido.shape, dtype=np.float32)
    centro = 254 + 24 * np.sin(y * .026) + 10 * np.cos(y * .044)
    distancia = np.abs(x - centro)
    ladera = 1 - np.exp(-(distancia / 45) ** 1.7)
    altura = ladera * (21 + 10 * ruido + 5 * np.abs(ruido)) + .38 * ruido
    dy, dx = np.gradient(altura)
    luz = np.clip((.72 * dx + .46 * dy + .45) / np.sqrt(1 + dx*dx + dy*dy), 0, 1)
    rio = np.exp(-(distancia / (2.7 + np.maximum(0, ruido)))**2)
    return altura, luz, rio, ruido


def ambiente(ruta, segundos, semilla):
    frecuencia = 24000
    t = np.arange(round(segundos * frecuencia)) / frecuencia
    base = np.zeros_like(t)
    brillo = np.zeros_like(t)
    for i, nota in enumerate((146.83, 220., 293.66, 329.63, 440., 587.33)):
        inicio = (0, .6, 1.65, 3.35, 4.9, 5.7)[i] * segundos / 8
        d = np.maximum(0, t-inicio)
        envolvente = (1-np.exp(-d*3)) * np.exp(-d*(.46 if i < 2 else .84))
        base += .13 * envolvente * np.sin(2*np.pi*nota*d + .35*np.sin(2*np.pi*(nota*1.5)*d))
        brillo += .022 * envolvente * np.sin(2*np.pi*(2*nota+.8)*d)
    for golpe in np.arange(.1, segundos-.7, .82):
        d = np.maximum(0, t-golpe)
        base += .12 * (t>golpe) * np.exp(-d*9) * np.sin(2*np.pi*48*d+9*(1-np.exp(-d*20)))
    viento = ndimage.gaussian_filter1d(np.random.default_rng(semilla).normal(size=len(t)), 3)
    aire = .03*viento*(.25+.75*np.sin(np.pi*t/segundos)**2)
    envolvente = np.clip(t/.28, 0, 1)*np.clip((segundos-t)/.7, 0, 1)
    stereo = np.column_stack((base+aire+brillo, base+aire+np.roll(brillo, 127)))
    with wave.open(str(ruta), "wb") as salida:
        salida.setparams((2, 2, frecuencia, 0, "NONE", ""))
        salida.writeframes((np.clip(stereo*envolvente[:,None], -1, 1)*32767).astype("<i2").tobytes())


def estelas(im, tiempo, color, ancho_logo):
    """Un haz recorre el valle y converge en dos órbitas alrededor de la marca."""
    w,h=im.size
    capa=Image.new("RGBA",(w,h))
    trazo=ImageDraw.Draw(capa)
    centro=(w*.5,h*.46)
    transicion=np.clip((tiempo-.32)/.36,0,1)
    transicion=transicion**2*(3-2*transicion)
    for j,tono in enumerate((tuple(map(int,color)),(255,237,172),(78,224,201))):
        fase=2*np.pi*(tiempo*1.10+j/3)
        u=np.linspace(0,1,88)
        ang=fase-u*2.7
        radio=ancho_logo*(.62+.035*j)
        orbita_x=centro[0]+radio*np.cos(ang)
        orbita_y=centro[1]+radio*(.75*np.sin(ang)+.16*np.cos(ang))
        v=tiempo+.43-u*.48
        camino_x=w*(.44+(.19+.09*u)*np.sin(v*9+j*.35))
        camino_y=h*(1.12-v*.94+.022*np.sin(u*9+j))
        xs=camino_x*(1-transicion)+orbita_x*transicion
        ys=camino_y*(1-transicion)+orbita_y*transicion
        intensidad=np.clip(tiempo/.12,0,1)*(.86-.44*transicion)
        for k in range(len(u)-1,0,-1):
            peso=(1-u[k])**.8
            alfa=round(220*intensidad*peso)
            linea=(round(xs[k]),round(ys[k]),round(xs[k-1]),round(ys[k-1]))
            trazo.line(linea,fill=(*tono,alfa),width=max(1,round((1.3+3*peso)*w/540)))
        r=2.7*w/540
        trazo.ellipse((xs[0]-r,ys[0]-r,xs[0]+r,ys[0]+r),fill=(*tono,round(245*intensidad)))
    halo=capa.filter(ImageFilter.GaussianBlur(max(2,w*.009)))
    im.paste(halo,mask=halo.getchannel("A"))
    im.paste(capa,mask=capa.getchannel("A"))


def renderizar(carpeta, logo, formato, direccion, *, silencio=False, segundos=8, avisar=None):
    """Recorrido con paralaje, luz, relieve y sonido sintetizado; cada cuadro se calcula en el tiempo."""
    identidad = hashlib.sha256(logo.read_bytes()).digest()
    semilla = int.from_bytes(identidad[:4], "big")
    altura, luz, rio, ruido = paisaje(semilla)
    w, h = (960,540) if formato == "horizontal" else (720,720) if formato == "cuadrado" else (540,960)
    fps = 24
    total = round(segundos*fps)
    tipo = direccion.get("paleta", "ambar")
    calido = np.array({"ambar": [255,153,54], "cobre":[255,116,73], "aurora":[95,224,205]}.get(tipo,[255,153,54]),dtype="f")
    cielo = np.array([8,22,46],dtype="f")
    yy,xx = np.mgrid[:h,:w].astype("f")
    nx,ny=xx/w,yy/h
    distancias=np.geomspace(4,265,245).astype("f")
    pantalla=np.linspace(-1,1,w,dtype="f")
    marca = recortar_logo(logo)
    audio = carpeta / "ambiente.wav"
    if not silencio:
        ambiente(audio, segundos, semilla)
    destino = carpeta / "campana.mp4"
    args = [ffmpeg(),"-y","-nostdin","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0"]
    if not silencio:
        args += ["-i",str(audio),"-c:a","aac","-b:a","128k"]
    args += ["-c:v","libx264","-threads","2","-crf","17","-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart",str(destino)]
    azar=np.random.default_rng(semilla)
    estrellas=azar.uniform(0,1,(150,3))
    motas=azar.uniform(0,1,(110,3))
    rapido = direccion.get("ritmo") == "ascendente"
    with subprocess.Popen(args, stdin=subprocess.PIPE) as escritor:
        try:
            for i in range(total):
                t=i/(total-1)
                avance=t*t*(3-2*t)
                z=68+(75 if rapido else 57)*avance
                camx=254+24*np.sin(z*.026)+10*np.cos(z*.044)+3*np.sin(4*t)
                horizonte=h*(.385+.067*avance)
                solx=.57+.08*np.sin(2.6*t)
                soly=horizonte/h-.025-.075*avance
                radio=np.sqrt(((nx-solx)*w/h)**2+(ny-soly)**2)
                atmosfera=np.exp(-radio*6)
                disco=np.clip((.030-radio)*h/2+.5,0,1)
                nube=ruido[((ny*75+95+3*t).astype(int))%512,((nx*140+45+9*t).astype(int))%512]
                resplandor=(atmosfera*(.63+.07*nube)+.22*np.exp(-np.abs(ny-horizonte/h)*8))
                img=cielo+(calido-cielo)*resplandor[:,:,None]+disco[:,:,None]*np.array([135,126,102])
                # Rayos contiguos por columna, de cerca a lejos. Solo se pintan los tramos visibles.
                mundo_x=camx+distancias[:,None]*(pantalla[None,:]*w/h*.90+.025*np.cos(3*t))
                mundo_z=np.broadcast_to(z+distancias[:,None],mundo_x.shape)
                coordenadas=[mundo_z,mundo_x]
                elev=ndimage.map_coordinates(altura,coordenadas,order=1,mode="wrap",prefilter=False)
                ilum=ndimage.map_coordinates(luz,coordenadas,order=1,mode="wrap",prefilter=False)
                agua=ndimage.map_coordinates(rio,coordenadas,order=1,mode="wrap",prefilter=False)
                camh=11+6*avance
                filas=np.clip(horizonte+(camh-elev)*h*.75/distancias[:,None],0,h).astype(int)
                vis=np.minimum.accumulate(filas,axis=0)
                # Buscar qué profundidad ocupa cada píxel en lugar de pintar polígonos Python.
                filas_pantalla=np.arange(h)[:,None]
                izq=np.zeros((h,w),np.int16);der=np.full((h,w),len(distancias),np.int16)
                columnas=np.arange(w)
                for _ in range(math.ceil(math.log2(len(distancias)+1))):
                    medio=(izq+der)//2
                    llega=vis[np.minimum(medio,len(distancias)-1),columnas]<=filas_pantalla
                    fin=izq==der
                    izq=np.where(fin|llega,izq,medio+1)
                    der=np.where(fin|~llega,der,medio)
                terreno=izq<len(distancias)
                capa=izq.clip(0,len(distancias)-1)
                dist=distancias[capa]
                niebla=np.exp(-dist/100.)
                l=ilum[capa,columnas]
                agua_px=agua[capa,columnas]
                destello=(.59+.34*np.sin(dist*.8-9*t+.8*np.sin(xx*.045)))*agua_px
                fondo=np.array([9,35,48])+l[:,:,None]**1.5*(calido*.64)
                fondo+=destello[:,:,None]*(calido*.9)
                horizonte_rgb=np.array([29,51,70])+(.25+.33*np.exp(-abs(nx-solx)*5))[:,:,None]*(calido-np.array([29,51,70]))
                img[terreno]=(fondo*niebla[:,:,None]+horizonte_rgb*(1-niebla[:,:,None]))[terreno]
                im=Image.fromarray(np.uint8(np.clip(img,0,255)))
                pincel=ImageDraw.Draw(im,"RGBA")
                for sx,sy,b in estrellas:
                    ey=int(sy*h*.36);ex=int((sx*w+4*t)%w)
                    if not terreno[ey,ex]:
                        pincel.ellipse((ex,ey,ex+1,ey+1),fill=(241,232,195,int(115*b*(1-.8*t))))
                for a,b,c in motas:
                    fase=(b+t*(.24+.32*a))%1
                    ex=int(w*(.5+(a-.5)*(.25+1.4*fase)+.05*np.sin(b*34+t*5)))
                    ey=int(h*(1.04-.69*fase+.10*np.sin(a*8+t*2)))
                    tam=1+int(2*c*fase);op=int(150*np.sin(np.pi*fase)**2)
                    pincel.ellipse((ex,ey,ex+tam,ey+tam),fill=(255,210,125,op))
                estelas(im,t,calido,min(w*.56,h*.43))
                aparicion=np.clip((t-.49)/.26,0,1)
                if aparicion:
                    mezcla=aparicion**2*(3-2*aparicion)
                    lado=round(min(w*.56,h*.43)*(.90+.10*mezcla))
                    arte=ImageOps.contain(marca,(lado,lado),Image.Resampling.LANCZOS)
                    xy=((w-arte.width)//2,round(h*(.46+.04*(1-mezcla))-arte.height/2))
                    im.paste(arte,xy,arte.getchannel("A").point(lambda a:round(a*mezcla)))
                escritor.stdin.write(im.tobytes())
                if avisar and i%fps==0:
                    avisar("componiendo",i,total)
            escritor.stdin.close()
            if escritor.wait(timeout=35):
                raise ValueError("Falló la exportación de la animación de marca.")
        except BaseException:
            escritor.kill()
            raise
    return destino
