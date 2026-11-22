"""Registro de una impresión con los detalles del video en coordenadas del recipiente."""
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as nd, signal

from estudio_medios import ejecutar, entrada_video

PASO = 88.
ACIMUT = -1.46
IDENTIDAD = np.array([[1., 0.], [0., 1.], [0., 0.]])


def borde(superficie, caja, ancho_estimado):
    """Buscar un borde superior casi recto que abarque la caja de tinta."""
    x, y, w, h = caja
    alto, ancho = superficie.shape[:2]
    a = max(0, int(y-max(.85*h,.41*ancho_estimado)))
    b = min(alto,int(y+max(h*.18,8)))
    if b-a<7:
        return None
    foto = superficie[a:b].astype(np.int16)
    r,g,az = foto.transpose(2,0,1)
    claro = (r>157)&(g>138)&(az>102)&(g-az<77)&(az>.48*r)
    etiquetas,_ = nd.label(nd.binary_closing(claro,structure=np.ones((2,5))))
    candidatos = []
    for k, seccion in enumerate(nd.find_objects(etiquetas),1):
        if seccion is None:
            continue
        ys,xs=seccion
        diametro=xs.stop-xs.start
        if not .66*ancho_estimado < diametro < 1.60*ancho_estimado:
            continue
        if not xs.start-w*.12 <= x+w*.5 <= xs.stop+w*.12:
            continue
        band=etiquetas[seccion]==k
        filas=np.flatnonzero(band.sum(axis=1)>.68*diametro)
        if not len(filas) or not .025*diametro<filas.size<.37*diametro:
            continue
        yy,xx=np.nonzero(band)
        cols=np.arange(int(.16*diametro),int(.84*diametro),3)
        superior=np.full(diametro,band.shape[0])
        np.minimum.at(superior,xx,yy)
        buenos=abs(superior[cols]-np.median(superior[cols]))<max(3,diametro*.029)
        if buenos.sum()<.62*cols.size:
            continue
        m,ordenada=np.polyfit(cols[buenos],superior[cols[buenos]],1)
        if abs(m)>.22:
            continue
        filas=filas[(filas>=filas[0]+1)&(filas<=filas[0]+max(4,filas.size*.45))]
        if not filas.size:
            continue
        cortes=band[filas]
        izq=xs.start+np.median(cortes.argmax(axis=1))
        der=xs.start+diametro-1-np.median(cortes[:,::-1].argmax(axis=1))
        radio=(der-izq)/2
        centro=(der+izq)/2
        cima=a+ys.start+ordenada+m*(centro-xs.start)
        if cima<2 or not .02*diametro<y+h*.5-cima<1.05*diametro:
            continue
        coste=abs(np.log(diametro/ancho_estimado))+.25*abs((y-cima)/max(1,h)-.30)
        candidatos.append((coste,[centro,radio,cima,m]))
    return min(candidatos,key=lambda par:par[0])[1] if candidatos else None


def material(forma, x, y):
    centro, radio, cima, m = forma
    dx,dy=x-centro,y-cima
    vertical=(dy-m*dx)/(1+m*m)
    horizontal=(dx+m*dy)/(1+m*m)
    rad=np.maximum(1,radio-.071*vertical)
    angulo=np.arcsin(np.clip(horizontal/rad,-1,1))
    return (angulo-ACIMUT)*PASO,vertical/radio*PASO,rad-abs(horizontal)


def vista(foto, forma, zs):
    centro, radio, cima, m=forma
    u,v=np.meshgrid(np.arange(259,dtype=np.float32),zs)
    y=v*radio/PASO
    x=(radio-.071*y)*np.sin(u/PASO+ACIMUT)
    coords=[cima+y+m*x,centro+x-m*y]
    # Los bordes de luminancia y de color responden tanto a las letras como al dibujo.
    ar=foto.astype(np.float32)/255
    valores=.35*ar[:,:,0]+.5*ar[:,:,1]+.15*ar[:,:,2]
    croma=np.ptp(ar,axis=2)
    grey=nd.map_coordinates(valores+.24*croma,coords,order=1,mode="nearest")
    sm=nd.gaussian_filter(grey,.7)
    media=nd.gaussian_filter(sm,4)
    desv=np.sqrt(np.maximum(.00055,nd.gaussian_filter(sm*sm,4)-media**2))
    return ((sm-media)/(desv+.035)).astype(np.float32)


def puntos(im, ventana):
    gy,gx=np.gradient(im)
    a,b,d=[nd.gaussian_filter(v,1.8) for v in (gx*gx,gx*gy,gy*gy)]
    fuerza=(a+d-np.sqrt((a-d)**2+4*b*b))*.5
    x,y,r,b=ventana
    yy,xx=np.indices(im.shape)
    admisible=(xx>max(x,12))&(xx<min(r,im.shape[1]-13))&(yy>max(y,10))&(yy<min(b,im.shape[0]-11))
    fuerza*=admisible
    # Separación espacial para que una sola letra no gobierne todo el registro.
    maximos=(nd.maximum_filter(fuerza,9)==fuerza)&(fuerza>.004)
    ys,xs=np.nonzero(maximos)
    orden=np.argsort(fuerza[ys,xs])[-45:]
    return np.column_stack((xs[orden],ys[orden]))


def buscar(ref, im, p, pred, margen):
    radio=6
    x,y=p
    xx,yy=np.rint(pred).astype(int)
    l=max(radio,xx-margen);r=min(im.shape[1]-radio-1,xx+margen)
    t=max(radio,yy-margen);b=min(im.shape[0]-radio-1,yy+margen)
    if r-l<4 or b-t<4:
        return None
    patron=ref[y-radio:y+radio+1,x-radio:x+radio+1]
    patron=patron-patron.mean()
    area=im[t-radio:b+radio+1,l-radio:r+radio+1]
    tam=patron.shape[0]
    corr=signal.correlate(area,patron,mode="valid",method="fft")
    def suma(z):
        integral=np.pad(z.cumsum(0,dtype=float).cumsum(1),((1,0),(1,0)))
        return integral[tam:,tam:]-integral[:-tam,tam:]-integral[tam:,:-tam]+integral[:-tam,:-tam]
    norma=np.maximum(1e-6,suma(area*area)-suma(area)**2/patron.size)
    corr/=np.sqrt(norma*max(.0001,(patron**2).sum()))
    v,u=np.unravel_index(corr.argmax(),corr.shape)
    if not (0<u<r-l and 0<v<b-t) or corr[v,u]<.56:
        return None
    def pico(a,b,c):
        return np.clip(.5*(a-c)/min(-1e-7,a+c-2*b),-.6,.6)
    return [l+u+pico(*corr[v,u-1:u+2]),t+v+pico(*corr[v-1:v+2,u]),corr[v,u]]


def ajustar(origen, destino, pesos, matriz):
    p,q,w=np.asarray(origen),np.asarray(destino),np.asarray(pesos)
    if len(p)<5:
        return matriz,0
    centro=np.median(p,axis=0)
    X=np.column_stack((p-centro,np.ones(len(p))))
    dif=q-(p@matriz[:2]+matriz[2])
    delta=np.median(dif,axis=0)
    base=matriz.copy()
    base[2]+=delta
    for umbral in (4.,2.5,1.65):
        error=np.linalg.norm(q-(p@base[:2]+base[2]),axis=1)
        peso=w/(1+(error/umbral)**4)
        reg=np.diag([27.,30.,.8])
        objetivo=np.vstack((matriz[:2],centro@matriz[:2]+matriz[2]+delta))
        coef=np.linalg.lstsq(np.vstack((X*peso[:,None],reg)),np.vstack((q*peso[:,None],reg@objetivo)),rcond=None)[0]
        base=np.vstack((coef[:2],coef[2]-centro@coef[:2]))
    buenos=np.linalg.norm(q-(p@base[:2]+base[2]),axis=1)<2.
    if buenos.sum()<5 or max(abs(base[:2]-np.eye(2)).ravel())>.30:
        return matriz,0
    return base,int(buenos.sum())


def rastrear(vistas, ventanas, ref):
    matrices=np.tile(IDENTIDAD,(len(vistas),1,1))
    muestras=[puntos(im,box) for im,box in zip(vistas,ventanas)]
    calidad=np.zeros(len(vistas),int)
    calidad[ref]=len(muestras[ref])
    for pasos in (range(ref-1,-1,-1), range(ref+1,len(vistas))):
        previo=ancla=ref
        velocidad=np.zeros(2)
        for i in pasos:
            base=matrices[previo].copy()
            base[2]+=velocidad
            pp=[];qq=[];ww=[]
            for k in dict.fromkeys((ref,ancla,previo)):
                canon=(muestras[k]-matrices[k,2])@np.linalg.inv(matrices[k,:2])
                pred=canon@base[:2]+base[2]
                for punto,ori,pos in zip(muestras[k],canon,pred):
                    m=buscar(vistas[k],vistas[i],punto,pos,17 if k==previo else 10)
                    if m:
                        pp.append(ori);qq.append(m[:2]);ww.append(m[2]**3*(1.8 if k==ref else 1))
            actual,calidad[i]=ajustar(pp,qq,ww,base)
            matrices[i]=actual
            eje=np.asarray([130.,vistas[i].shape[0]*.5])
            movimiento=eje@(actual[:2]-matrices[previo,:2])+actual[2]-matrices[previo,2]
            velocidad=np.clip(.70*movimiento+.30*velocidad,-16,16) if calidad[i] else velocidad*.35
            if calidad[i]>=9 and abs(i-ancla)>=6:
                ancla=i
            previo=i
    return matrices,calidad


def mascara_referencia(frames, matrices, formas, cajas, vistas, calidad, z0, rectangulo):
    # En el atlas del vaso la tinta permanece fija; burbujas, reflejos y manos
    # cambian entre muestras. Conservar los trazos que coinciden en varias vistas.
    yy,xx=np.indices(vistas[0].shape,dtype=np.float32)
    uv=np.stack((xx,yy),axis=2)
    l,t,r,b=rectangulo
    h,w=b-t,r-l
    dentro=(xx>l-.10*w)&(xx<r+.13*w)&(yy>t-.07*h)&(yy<b+.09*h)
    centro=(xx>l+.06*w)&(xx<r-.06*w)&(yy>t+.06*h)&(yy<b-.09*h)
    marco=(~dentro)&(yy>t+.03*h)&(yy<b+.20*h)
    if not centro.any() or not marco.any():
        return None
    muestras=np.flatnonzero((calidad>=7)&(cajas[:,2]>=.72*np.percentile(cajas[:,2],85)))
    if not muestras.size:
        return None
    indices=np.unique(muestras[np.linspace(0,len(muestras)-1,min(11,len(muestras))).astype(int)])
    colores=[]
    for i in indices:
        coords=uv@matrices[i,:2]+matrices[i,2]
        angulo=ACIMUT+coords[:,:,0]/PASO
        eje,radio,cima,inclinacion=formas[i]
        dy=(coords[:,:,1]+z0)*radio/PASO
        dx=(radio-.071*dy)*np.sin(angulo)
        mx=eje+dx-inclinacion*dy;my=cima+dy+inclinacion*dx
        rgb=np.stack([nd.map_coordinates(frames[i,:,:,c],[my,mx],order=1,mode="nearest") for c in range(3)],2)
        colores.append(np.asarray(Image.fromarray(rgb).convert("HSV"),dtype=np.int16))
    hsv=np.stack(colores)
    tono,sat,val=hsv.transpose(3,0,1,2)
    vivos=(sat>90)&(val>88)
    def hist(zona):
        conteo=np.bincount(tono[vivos&zona],minlength=256).astype(float)
        return nd.gaussian_filter1d(conteo,2,mode="wrap")/(len(indices)*zona.sum())
    interior,exterior=hist(centro),hist(marco)
    elegido=np.argmax(interior/(.0005+exterior)**.8)
    if interior[elegido]<.002 or interior[elegido]<1.7*exterior[elegido]:
        return None
    diferencia=abs((tono-elegido+128)%256-128)
    principal=np.mean(vivos&(diferencia<7),axis=0)>.46
    principal=nd.median_filter((principal&dentro).astype(np.uint8),size=3).astype(bool)
    if not principal.any():
        return None
    grupos,_=nd.label(nd.maximum_filter(principal,size=3))
    areas=np.bincount(grupos.ravel())[1:]
    principal &= np.isin(grupos,np.flatnonzero(areas>=max(5,.038*areas.max()))+1)
    cerca=nd.distance_transform_edt(~principal)<.36*h
    blancos=np.mean((sat<106)&(val>145),axis=0)>.49
    oscuros=np.mean(val<95,axis=0)>.47
    # El límite superior descarta la espuma incluso cuando ocupa parte de la caja inicial.
    filas=np.where(principal.any(axis=1))[0]
    if not len(filas):
        return None
    blancos &= yy>filas[0]-.055*h
    base=principal|(cerca&blancos&dentro)
    borde=nd.distance_transform_edt(~base)<max(3,h*.072)
    fijo=(base|(borde&oscuros))&dentro
    return nd.gaussian_filter(nd.maximum_filter(fijo.astype(np.float32),size=max(3,round(h*.079))|1),.85)


class Registro:
    def __init__(self, tiempos, estados, rectangulo, mascara):
        self.tiempos=tiempos
        self.estados=estados
        self.rectangulo=rectangulo
        self.mascara=mascara

    def en(self,t):
        if not self.tiempos[0]-.068<=t<=self.tiempos[-1]+.068:
            return None
        j=np.clip(np.searchsorted(self.tiempos,t),1,len(self.tiempos)-1)
        f=np.clip((t-self.tiempos[j-1])/(self.tiempos[j]-self.tiempos[j-1]),0,1)
        estado=(1-f)*self.estados[j-1]+f*self.estados[j]
        return estado[:4],estado[4:].reshape(3,2),self.rectangulo,self.mascara


def registrar(frames, tiempos, cajas, escalas):
    sx,sy=escalas
    caja=cajas*np.array([sx,sy,sx,sy])
    w,h=caja[:,2],caja[:,3]
    anchos=w[w>=np.percentile(w,60)]
    diametro=max(np.median(anchos)*1.36,np.percentile(h,70)*1.51)
    formas=[borde(im,c,diametro) for im,c in zip(frames,caja)]
    buenas=np.array([g is not None for g in formas])
    if buenas.sum()<max(5,len(formas)*.86):
        return None
    idx=np.flatnonzero(buenas)
    gs=np.asarray([formas[i] for i in idx])
    geom=np.column_stack([np.interp(np.arange(len(frames)),idx,gs[:,j]) for j in range(4)])
    geom=nd.gaussian_filter1d(nd.median_filter(geom,size=(3,1),mode="nearest"),.48,axis=0,mode="nearest")
    frontales=np.flatnonzero((w/h>=np.percentile(w/h,62))&buenas)
    ref=frontales[len(frontales)//2]
    ventanas=[]
    for (x,y,ww,hh),g in zip(caja,geom):
        l,t,_=material(g,x,y)
        r,b,_=material(g,x+ww,y+hh)
        ventanas.append([l,t,r,b])
    ventanas=np.asarray(ventanas)
    z0=math.floor(np.percentile(ventanas[:,1],8)-13)
    z1=min(z0+236,math.ceil(np.percentile(ventanas[:,3],90)+14))
    if z1-z0<22:
        return None
    zs=np.arange(z0,z1,dtype=np.float32)
    ventanas[:,[1,3]]-=z0
    ims=[vista(f,g,zs) for f,g in zip(frames,geom)]
    matrices,calidad=rastrear(ims,ventanas,ref)
    if (calidad>=6).sum()<len(calidad)*.68:
        return None
    # Interpolar solo huecos internos breves. Fuera de una racha comprobada se usa la caja habitual.
    aceptadas=np.flatnonzero(calidad>=5)
    tramo=np.split(aceptadas,np.flatnonzero(np.diff(aceptadas)>4)+1)
    continuo=next(g for g in tramo if ref in g) if ref in aceptadas else max(tramo,key=len)
    if len(continuo)<4:
        return None
    ini,fin=continuo[0],continuo[-1]+1
    original=geom.copy()
    original[:,[0,1]]/=sx
    original[:,2]/=sy
    original[:,3]*=sx/sy
    estados=np.column_stack((original,matrices.reshape(-1,6)))
    rect=tuple(ventanas[ref])+(z0,)
    mascara=mascara_referencia(frames,matrices,geom,caja,ims,calidad,z0,ventanas[ref])
    return Registro(tiempos[ini:fin],estados[ini:fin],rect,mascara)


def preparar_registros(clip, grupos, region):
    x,y,w,h=region
    escala=min(1,420/max(w,h))
    sw,sh=max(4,round(w*escala)),max(4,round(h*escala))
    tramo=grupos[-1][1]-grupos[0][0][0][0]
    fps=min(15,max(6,math.floor(400/max(1,tramo))))
    inicio=math.floor(grupos[0][0][0][0]*fps)/fps
    duracion=grupos[-1][1]-inicio
    with tempfile.TemporaryDirectory(dir=Path(clip).parent) as temp:
        ruta=Path(temp)/"superficie.rgb"
        filtro=f"fps={fps},crop={w}:{h}:{x}:{y}:exact=1,scale={sw}:{sh}"
        ejecutar(["-y","-v","error","-ss",str(inicio),*entrada_video(clip),"-map","0:v:0","-an","-t",str(duracion),
                  "-vf",filtro,"-threads","1","-fps_mode","passthrough","-pix_fmt","rgb24","-f","rawvideo",str(ruta)],tiempo=80)
        cantidad=ruta.stat().st_size//(sw*sh*3)
        if cantidad<5:
            return [None]*len(grupos)
        frames=np.memmap(ruta,dtype="uint8",mode="r",shape=(cantidad,sh,sw,3))
        tiempos=inicio+np.arange(cantidad)/fps
        registros=[]
        for grupo,fin in grupos:
            puntos=np.asarray(grupo)
            idx=np.flatnonzero((tiempos>=puntos[0,0]-1e-7)&(tiempos<fin))
            if len(idx)<5:
                registros.append(None)
                continue
            ts=tiempos[idx]
            cajas=np.column_stack([np.interp(ts,puntos[:,0],puntos[:,k]) for k in range(1,5)])
            cajas[:,:2]-=[x,y]
            registros.append(registrar(frames[idx[0]:idx[-1]+1],ts,cajas,(sw/w,sh/h)))
        return registros


ARCHIVO_REGISTRO = "registro-superficie.npz"


def registros_del_clip(clip, grupos, region):
    """Reutilizar las mediciones del original al cambiar únicamente la impresión."""
    firma = hashlib.sha256(json.dumps([1, grupos, region], separators=(",", ":")).encode())
    with clip.open("rb") as fuente:
        for bloque in iter(lambda: fuente.read(1024 * 1024), b""):
            firma.update(bloque)
    clave = firma.hexdigest()
    cache = clip.parent / ARCHIVO_REGISTRO
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as datos:
                if datos["clave"].item() == clave:
                    registros = []
                    for i in range(len(grupos)):
                        if not datos["presentes"][i]:
                            registros.append(None)
                            continue
                        mascara = datos[f"mascara{i}"]
                        registros.append(Registro(datos[f"tiempos{i}"], datos[f"estados{i}"],
                                                  datos[f"rectangulo{i}"], mascara if mascara.size else None))
                    return registros
        except (OSError, ValueError, EOFError, IndexError, KeyError, zipfile.BadZipFile):
            pass
    registros = preparar_registros(clip, grupos, region)
    campos = {"clave": clave, "presentes": [r is not None for r in registros]}
    for i, registro in enumerate(registros):
        if registro is None:
            continue
        campos[f"tiempos{i}"] = registro.tiempos
        campos[f"estados{i}"] = registro.estados
        campos[f"rectangulo{i}"] = registro.rectangulo
        campos[f"mascara{i}"] = registro.mascara if registro.mascara is not None else np.empty(0, np.float32)
    # Reemplazo atómico: el registro heredado de la revisión anterior es inmutable.
    try:
        with tempfile.TemporaryDirectory(dir=clip.parent) as temporal:
            nuevo = Path(temporal) / ARCHIVO_REGISTRO
            np.savez_compressed(nuevo, **campos)
            nuevo.replace(cache)
    except OSError:
        pass  # La caché es opcional; las mediciones siguen en memoria para este render.
    return registros


def proyectar(estado, x, y, proporcion):
    forma, matriz, (l,t,r,b,z0), mascara=estado
    a,z,distancia=material(forma,x,y)
    uv=np.stack((a,z-z0),axis=2)
    uv=(uv-matriz[2])@np.linalg.inv(matriz[:2])
    ancho,alto=max(r-l,1),max(b-t,1)
    h=min(alto,ancho/proporcion)
    u=(uv[:,:,0]-(l+r)/2)/(h*proporcion)+.5
    v=(uv[:,:,1]-(t+b)/2)/h+.5
    anterior=None if mascara is None else nd.map_coordinates(mascara,[uv[:,:,1],uv[:,:,0]],order=1,mode="constant")
    return u,v,np.clip(distancia/1.7,0,1),anterior
