"""Impresión sobre una superficie del video, siguiendo la trayectoria del diseño anterior."""
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


def acabado_impreso(texto):
    """La última indicación de acabado tiene prioridad en revisiones de un reel."""
    for pedido in reversed(texto.lower().split("ajuste solicitado:")):
        if re.search(r"marca de agua|serigraf|impres[oa]|imprim|estampad|grabado|sin (?:fondo|pegatina|sticker)", pedido):
            return True
        if re.search(r"pegatina|sticker|calcoman[ií]a", pedido):
            return False
        if re.search(r"\b(?:vaso|vidrio|cristal|copa|botella|taza)\b", pedido):
            return True
    return False


def preparar_tinta(ruta):
    with Image.open(ruta) as imagen:
        arte = ImageOps.exif_transpose(imagen).convert("RGBA")
        arte.thumbnail((768, 768), Image.Resampling.LANCZOS)
    colores = np.asarray(arte).astype(np.float32) / 255
    rgb, alfa = colores[:, :, :3], colores[:, :, 3]
    # Conservar el canal alfa de un PNG. Para un JPG sobre papel claro, separar
    # también los blancos centrales para que se vea el material entre las letras.
    borde = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    if alfa.min() > .97 and np.mean(borde.min(axis=1) > .83) > .70:
        intensidad = np.clip((.78-rgb.mean(axis=2))/.26, 0, 1)
        croma = np.clip((np.ptp(rgb, axis=2)-.065)/.15, 0, 1)
        alfa = np.maximum(intensidad, croma)
    return np.dstack((rgb * alfa[:, :, None], alfa))


def vista_frontal(grupo):
    cajas = np.asarray(grupo)
    formas = cajas[:, 3] / cajas[:, 4]
    frontal = max(.01, np.percentile(formas, 95))
    perfil, frente = cajas[np.argmin(formas)], cajas[np.argmax(formas)]
    signo = 1 if abs(perfil[1]-frente[1]) >= abs((perfil[1]+perfil[3])-(frente[1]+frente[3])) else -1
    return frontal, signo


def coordenadas_arte(u, v, aspecto, vista, curva, proporcion):
    if curva:
        frontal, signo = vista
        factor = np.clip(aspecto/frontal, .001, 1)
        arco = .84
        lo, hi = 0., math.pi/2+arco
        for _ in range(13):
            medio = (lo+hi)/2
            ancho = (math.sin(min(math.pi/2, medio+arco))-math.sin(medio-arco))/(2*math.sin(arco))
            if ancho > factor:
                lo = medio
            else:
                hi = medio
        angulo = (hi+lo)/2
        izquierda = math.sin(angulo-arco)
        derecha = math.sin(min(math.pi/2, angulo+arco))
        posicion = u if signo > 0 else 1-u
        phi = np.arcsin(np.clip(izquierda+posicion*(derecha-izquierda), -1, 1))
        fase = (phi-angulo+arco)/(2*arco)
        x = fase if signo > 0 else 1-fase
        y = v + .022*(np.cos(phi)-math.cos(angulo))
        normal = np.maximum(0, np.cos(phi))
    else:
        frontal, x, y, normal = aspecto, u, v, np.ones_like(u)
    # Ajustar el arte completo al espacio de la serigrafía.
    return ((x-.5)*max(1, frontal/proporcion)+.5,
            (y-.5)*max(1, proporcion/frontal)+.5, normal)


def analizar_soporte(foto, u, v, curva, signo):
    forma = np.ones(u.shape, dtype=np.float32)
    if not curva:
        return forma, forma
    hsv = np.asarray(Image.fromarray(foto).convert("HSV")).astype(np.float32)
    matiz, sat, valor = hsv.transpose(2,0,1)
    muestra = (u>.14)&(u<.84)&(v>1.065)&(v<1.38)
    cromatico = muestra & (sat>60) & (valor>36)
    if cromatico.sum() < max(8, .42*muestra.sum()):
        return forma, forma
    hist = np.bincount(matiz[cromatico].astype(int), minlength=256)
    hist = ndimage.gaussian_filter1d(hist.astype(float), 3, mode="wrap")
    central = hist.argmax()
    delta = abs((matiz-central+128)%256-128)
    if np.mean(delta[cromatico]<14) < .52:
        return forma, forma
    sat_min = max(40, .66*np.percentile(sat[cromatico],20))
    afinidad = np.clip((16-delta)/9,0,1)*np.clip((sat-sat_min)/48,0,1)
    # Cerca de una silueta vertical, reconocer la transición desde el color del
    # soporte hasta el fondo. Examinar varias inclinaciones sin desplazar el arte.
    yy, xx = np.indices(u.shape)
    pix = foto.astype(np.float32)/255
    h = max(1, np.count_nonzero((v[:,0] >= 0) & (v[:,0] <= 1)))
    w = max(1, np.count_nonzero((u[0]>=0)&(u[0]<=1)))
    ycentro = np.argmin(abs(v[:,0]-.5))
    filas = np.flatnonzero((v[:,0]>.02)&(v[:,0]<1.30))
    paso = max(2, round(h*.023))
    if len(filas)<5:
        return afinidad, forma
    muestra_rgb = np.median(pix[cromatico],axis=0)
    eje = muestra_rgb - muestra_rgb.mean()
    eje /= max(.01,np.linalg.norm(eje))
    croma = pix @ eje
    for lado in (signo,):
        ancla = (0 if lado==-1 else 1)
        cols = np.flatnonzero((abs(u[0]-ancla)<.22+5/w) & (xx[0]>=paso+2)&(xx[0]<pix.shape[1]-paso-2))
        if cols.size == 0:
            continue
        mejor = None
        for pendiente in (-.14,-.09,-.04,0,.04,.09,.14):
            xs = np.rint(cols[None,:]+(filas[:,None]-ycentro)*pendiente).astype(int)
            xs = np.clip(xs,paso,pix.shape[1]-paso-1)
            ys = filas[:,None]
            dentro, fuera = xs-lado*paso, xs+lado*paso
            cambio = croma[ys,dentro]-croma[ys,fuera]
            grad = np.sqrt(np.mean((pix[ys,dentro]-pix[ys,fuera])**2,axis=2))
            contraste = grad*np.clip(croma[ys,dentro]*4,.09,1)
            voto = np.mean(contraste+.88*np.clip(cambio,-.4,.65),axis=0)
            j = voto.argmax()
            if mejor is None or voto[j]>mejor[0]:
                mejor = (voto[j],cols[j],pendiente)
        puntuacion, columna, pendiente = mejor
        if puntuacion > .20:
            limite = columna+(yy-ycentro)*pendiente
            forma *= np.clip(.4+lado*(limite-xx)/1.5,0,1)
    # Mantener el borde de una franja blanca que atraviesa el recipiente
    # (espuma o reflejo); no confundirla con la tinta clara de un solo trazo.
    centro = (u>.1)&(u<.85)&(forma>.5)
    neutrales = (sat<92)&(valor>186)&centro
    franja = neutrales.sum(axis=1) > .53*np.maximum(1,centro.sum(axis=1))
    superiores = np.flatnonzero(franja & (v[:,0]<.23) & (v[:,0]>-.32))
    if len(superiores):
        forma *= np.clip((yy-superiores[-1]-.5)/2,0,1)
    return afinidad, forma


def extension_tinta(foto, u, v, alto, afinidad, limite):
    interior = (u>=-.055) & (u<=1.09) & (v>=-.035) & (v<=1.035) & (limite>.6)
    entorno = (u>-.22) & (u<1.22) & (v>-.30) & (v<1.38) & ~interior
    if not np.any(entorno):
        return interior
    hsv = np.asarray(Image.fromarray(foto).convert("HSV"), dtype=np.int32)
    # Suprimir la dependencia del matiz en los reflejos de baja saturación.
    tono = np.where(hsv[:, :, 1] < 48, 24, hsv[:, :, 0]*24//256)
    codigo = (tono*5+hsv[:, :, 1]*5//256)*5+hsv[:, :, 2]*5//256
    def hist(zona):
        return np.bincount(codigo[zona], minlength=625)/max(1, zona.sum())
    dentro, fuera = hist(interior), hist(entorno)
    tinta = interior & (dentro[codigo] > 2.4*fuera[codigo]+.0015)
    if np.any(afinidad < 1):
        # Identificar la familia de color de la tinta por su exceso respecto al entorno.
        # Los trazos claros y negros adyacentes pertenecen al mismo diseño.
        hue, sat, val = hsv.transpose(2,0,1)
        vivos = (sat>78)&(val>81)
        def distribucion(zona):
            conteo = np.bincount(hue[zona & vivos],minlength=256).astype(float)
            return ndimage.gaussian_filter1d(conteo,2,mode="wrap")/max(1,zona.sum())
        contraste = distribucion(interior)*(1/(.0009+distribucion(entorno)))**.72
        centro = contraste.argmax()
        diferencia = np.abs((hue-centro+128)%256-128)
        principal = (diferencia<9)&vivos&(afinidad<.62)&interior
        if principal.sum()>max(8,.01*interior.sum()):
            # Las manchas aisladas del fondo no sirven como punto de partida.
            etiquetas, _ = ndimage.label(ndimage.maximum_filter(principal,size=max(3,round(alto*.025))))
            areas = np.bincount(etiquetas.ravel())[1:]
            principal &= np.isin(etiquetas,np.flatnonzero(areas>=max(6,.07*areas.max()))+1)
            distancia = ndimage.distance_transform_edt(~principal)
            cerca = distancia<alto*.63
            claro = (sat<94)&(val>130)
            oscuro = (val<95)&(afinidad<.78)
            lateral = (u>-.07)&(u<1.10)&(v>-.035)&(v<1.055)
            # Una franja amplia de espuma o luz queda fuera de las letras.
            blanco = (sat<95)&(val>187)&interior
            banda = blanco.sum(axis=1) > .45*np.maximum(1,interior.sum(axis=1))
            tinta_clara = cerca & claro & ~banda[:,None]
            opaco = principal | tinta_clara | (cerca&(diferencia<12)&(afinidad<.28)&(val>93))
            filo = oscuro & (ndimage.distance_transform_edt(~opaco)<max(4,alto*.046))
            trazo = (opaco | filo) & lateral & (limite>.4)
            radio = max(2,round(alto*.036))
            return ndimage.maximum_filter(trazo,size=radio*2+1)>0
    radio = max(1, round(alto*.035))
    tinta = ndimage.median_filter(tinta.astype(np.uint8), size=3).astype(bool)
    filas, columnas = np.nonzero(tinta)
    if len(filas) < max(6, .012*interior.sum()):
        return interior
    # Unir las letras y líneas del mismo dibujo sin crear un parche rectangular.
    extremos = []
    for y in np.unique(filas):
        xs = columnas[filas == y]
        extremos.extend(((xs.min(), y), (xs.max(), y)))
    puntos = np.array(extremos)
    try:
        vertices = puntos[ConvexHull(puntos).vertices].tolist()
    except QhullError:
        return interior
    contorno = Image.new("L", (foto.shape[1], foto.shape[0]))
    ImageDraw.Draw(contorno).polygon([tuple(p) for p in vertices], fill=1)
    return ndimage.maximum_filter(np.asarray(contorno), size=2*radio+1) > 0


def reconstruir(foto, mascara, alto, afinidad, limite):
    # Relleno multiescala con muestras del objeto que rodean la tinta anterior.
    a = foto.astype(np.float32)/255
    reducido = Image.Resampling.BILINEAR
    fuente = np.asarray(Image.fromarray(foto).resize((max(1,foto.shape[1]//3), max(1,foto.shape[0]//3)), reducido), dtype=np.float32)/255
    libres = (~ndimage.maximum_filter(mascara, size=5)).astype(np.float32)*limite*afinidad
    if libres.sum()<8:
        libres = (~mascara).astype(np.float32)
    peso = np.asarray(Image.fromarray(libres).resize((fuente.shape[1], fuente.shape[0]), reducido))
    if peso.max() == 0:
        return a
    promedio = np.sum(fuente*peso[:,:,None],axis=(0,1))/max(1e-8,peso.sum())
    fondo = np.broadcast_to(promedio,fuente.shape)
    for sigma in (max(4, alto*.22), max(3, alto*.085), max(1.5, alto*.033)):
        cuenta = ndimage.gaussian_filter(peso, sigma)
        color = ndimage.gaussian_filter(fuente*peso[:,:,None], (sigma,sigma,0))
        mezcla = np.clip(cuenta/.048, 0, 1)[:,:,None]
        fondo = fondo*(1-mezcla) + mezcla*color/np.maximum(cuenta[:,:,None], 1e-8)
    fondo = np.asarray(Image.fromarray(np.uint8(np.clip(fondo*255,0,255))).resize((foto.shape[1],foto.shape[0]), reducido),dtype=np.float32)/255
    # Devolver textura fina de la superficie limpia más próxima.
    ys = np.arange(a.shape[0])
    filas = np.where(mascara.any(axis=1))[0]
    referencia = filas[-1] if len(filas) else a.shape[0]//2
    origen = np.clip(referencia+3+.60*abs(referencia-ys),0,a.shape[0]-1).astype(int)
    desplazado = a[origen]
    grano = desplazado-ndimage.gaussian_filter(desplazado,(2,2,0))
    return np.clip(fondo+.52*np.clip(grano, -.08, .08), 0, 1)


def imprimir(fotograma, caja, arte, vista, curva):
    x, y, w, h = caja
    margen = max(6, round(max(w,h)*.34))
    region = (max(0,x-margen), max(0,y-margen), min(fotograma.width,x+w+margen), min(fotograma.height,y+h+margen))
    recorte = fotograma.crop(region)
    original = recorte.size
    recorte.thumbnail((432,432),Image.Resampling.LANCZOS)
    fx, fy = recorte.width/original[0], recorte.height/original[1]
    yy, xx = np.indices((recorte.height,recorte.width), dtype=np.float32)
    u = ((xx+.5)/fx + region[0]-x)/w
    v = ((yy+.5)/fy + region[1]-y)/h
    alto = h*fy
    foto = np.asarray(recorte)
    afinidad, limite = analizar_soporte(foto,u,v,curva,vista[1])
    mascara = extension_tinta(foto, u, v, alto, afinidad, limite)
    opacidad = ndimage.gaussian_filter(mascara.astype(np.float32), max(.65, alto*.014))
    if np.any(afinidad<1):
        # Conservar el vidrio que ya se ve entre tintas y suavizar la transición
        # alrededor de los trazos que sí es necesario reconstruir.
        intacto = ndimage.minimum_filter(afinidad, size=max(3,round(alto*.035)))
        opacidad *= 1-.98*np.clip((intacto-.55)/.44,0,1)
    opacidad *= limite
    muestras = afinidad * np.clip((v+.025)/.08,0,1)
    fondo = reconstruir(foto, mascara, alto, muestras, limite)
    original_rgb = foto.astype(np.float32)/255
    base = original_rgb*(1-opacidad[:,:,None])+fondo*opacidad[:,:,None]
    px, py, normal = coordenadas_arte(u, v, w/h, vista, curva, arte.shape[1]/arte.shape[0])
    coords = [py*(arte.shape[0]-1), px*(arte.shape[1]-1)]
    estampado = np.stack([ndimage.map_coordinates(arte[:,:,c], coords, order=1, mode="constant") for c in range(4)], 2)
    visible = (u>=0)&(u<=1)&(v>=0)&(v<=1)
    densidad = .82*(.66+.34*normal)*visible*limite
    luz = np.mean(base, axis=2)
    tinta = estampado[:,:,:3]*(.73+.38*luz[:,:,None])
    alfa = estampado[:,:,3]*densidad
    suave = max(.50, min(1.25, alto/270))
    alfa = ndimage.gaussian_filter(alfa, suave)
    color = ndimage.gaussian_filter(tinta*densidad[:,:,None], (suave,suave,0))
    resultado = base*(1-alfa[:,:,None]) + color
    # Solo se deben escribir los píxeles afectados, también en cuadros de tasa variable.
    activo = (opacidad+alfa) > .004
    rgba = np.dstack([np.uint8(np.clip(resultado*255+.5,0,255)), activo.astype(np.uint8)*255])
    rgba[~activo] = 0
    capa = Image.fromarray(rgba)
    return capa.resize(original,Image.Resampling.BILINEAR), region[:2]
