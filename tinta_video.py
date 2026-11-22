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
    espacio = max(0, a.shape[0]-referencia-3)
    distancia = np.clip((referencia-ys)/max(1, alto),0,1)
    origen = np.clip(referencia+2+distancia*min(espacio,alto*.70),0,a.shape[0]-1).astype(int)
    desplazado = a[origen]
    grano = (desplazado-ndimage.gaussian_filter(desplazado,(2.6,2.6,0)))*min(1,espacio/12)
    return np.clip(fondo+.52*np.clip(grano, -.08, .08), 0, 1)


def superficie_limpia(foto, u, v, alto, mascara, afinidad, limite):
    """Separar tinta de material por la cromaticidad observada fuera de la impresión."""
    rgb = foto.astype(np.float32)/255
    total = rgb.sum(2)
    sat = np.ptp(rgb,axis=2)/np.maximum(.03,rgb.max(2))
    candidatos = ((v>1.09)&(v<1.7)&(u>.14)&(u<.86)&(limite>.9)&
                  (sat>.32)&(total>.24)&(afinidad>.58))
    if candidatos.sum()<30:
        return mascara, None
    # El percentil ignora las burbujas y los reflejos especulares de la muestra.
    hsv=np.asarray(Image.fromarray(foto).convert("HSV"),dtype=np.float32)
    tono=hsv[:,:,0]
    hist=ndimage.gaussian_filter1d(np.bincount(tono[candidatos].astype(int),minlength=256).astype(float),2,mode="wrap")
    distancia=np.abs((tono-hist.argmax()+128)%256-128)
    if np.mean(distancia[candidatos]<15)<.65:
        return mascara,None
    tolerancia=float(np.clip(np.percentile(distancia[candidatos],58)+2.5,3,8))
    # La tinta antigua ocupa una silueta común, pero deja huecos de vidrio limpio.
    # Unir sus componentes aquí evita que las letras reaparezcan al mover el arte.
    filas,cols=np.nonzero(mascara)
    if len(filas)<4:
        return mascara,None
    puntos=np.column_stack((cols,filas))
    try:
        vertices=puntos[ConvexHull(puntos).vertices]
    except QhullError:
        return mascara,None
    silueta=Image.new("L",(foto.shape[1],foto.shape[0]))
    ImageDraw.Draw(silueta).polygon([tuple(p) for p in vertices.tolist()],fill=1)
    silueta=ndimage.maximum_filter(np.asarray(silueta),size=max(5,round(alto*.14))|1)>0
    cerca = (u>-.10)&(u<1.12)&(v>-.08)&(v<1.10)&(limite>.4)
    limpio = np.clip((tolerancia*1.20-distancia)/max(.015,tolerancia*.45),0,1)
    limpio *= np.clip(total/.22,0,1)*np.clip((sat-.28)/.30,0,1)
    intacto = ndimage.minimum_filter(limpio,size=max(3,round(alto*.022))|1)
    campo = silueta & cerca
    saturada=(sat>.35)&(rgb.max(2)>.30)
    pigmentos=campo & saturada & (distancia>tolerancia*1.45)
    if pigmentos.sum()<max(10,.015*campo.sum()):
        return mascara,None
    hist_tinta=ndimage.gaussian_filter1d(np.bincount(tono[pigmentos].astype(int),minlength=256).astype(float),2,mode="wrap")
    delta=np.abs((tono-hist_tinta.argmax()+128)%256-128)
    color_tinta=campo & saturada & (delta<11) & (distancia>tolerancia*.76)
    cerca_color=ndimage.distance_transform_edt(~color_tinta)<max(5,alto*.54)
    primer_trazo=np.min(np.where(color_tinta)[0]) if color_tinta.any() else 0
    bajo_espuma = np.arange(foto.shape[0])[:,None] >= primer_trazo-alto*.07
    claros=(sat<.42)&(rgb.max(2)>.51)
    tinta=color_tinta|(campo & claros & cerca_color & bajo_espuma)
    contorno=(rgb.max(2)<.40)&(ndimage.distance_transform_edt(~tinta)<max(4,alto*.065))
    tinta |= campo & contorno
    tinta=ndimage.maximum_filter(tinta,size=max(5,round(alto*.060))|1)
    return tinta & (limite>.35),intacto



def muestra_de_textura(fotograma, caja):
    """Elegir una muestra bidimensional bajo la impresión en una vista frontal."""
    x,y,w,h=caja
    lado=max(12,round(min(w,h)*.54))
    mejor, puntuacion=None,0.
    for fx in (.15,.36,.57):
        for fy in (1.10,1.34):
            x0,y0=round(x+fx*w),round(y+fy*h)
            if x0<0 or y0<0 or x0+lado>fotograma.width or y0+lado>fotograma.height:
                continue
            muestra=fotograma.crop((x0,y0,x0+lado,y0+lado)).resize((80,80),Image.Resampling.BILINEAR)
            rgb=np.asarray(muestra,dtype=np.float32)/255
            luminancia=rgb@np.array([.30,.57,.13],dtype=np.float32)
            detalle=luminancia-ndimage.gaussian_filter(luminancia,3.8)
            gy,gx=np.gradient(luminancia)
            # Penalizar un solo filo fuerte o una fuente quemada; preferir grano repartido.
            matriz=np.array([[np.mean(gx*gx),np.mean(gx*gy)],[np.mean(gx*gy),np.mean(gy*gy)]])
            menor,mayor=np.linalg.eigvalsh(matriz)
            energia=np.percentile(np.abs(detalle),78)
            nota=energia*max(0,menor)/(mayor+.00001)*np.mean((luminancia>.07)&(luminancia<.88))
            if nota>puntuacion and energia>.006:
                puntuacion,mejor=nota,np.clip(detalle,-.10,.15)
    return mejor


def textura_de_superficie(foto, mascara, afinidad, limite, alto):
    """Transportar detalle de zonas limpias sin estirar filas sobre la marca."""
    rgb = foto.astype(np.float32)/255
    sigma = max(1.2,min(4.,alto*.024))
    detalle = rgb - ndimage.gaussian_filter(rgb,(sigma,sigma,0))
    libre = (~ndimage.maximum_filter(mascara,size=5))*afinidad*limite
    confianza = ndimage.uniform_filter(libre.astype(np.float32),size=max(5,round(sigma*4))|1)
    if libre.sum()<8:
        return reconstruir(foto,mascara,alto,afinidad,limite)
    # Reducir el color premultiplicado por la confianza; así la tinta vecina
    # no se filtra en las muestras limpias al bajar o subir la resolución.
    paso = 3
    pesos = ndimage.uniform_filter(libre,size=paso)[::paso,::paso]
    color = ndimage.uniform_filter(rgb*libre[:,:,None],size=(paso,paso,1))[::paso,::paso]
    validos=pesos>.08
    if not validos.any():
        return reconstruir(foto,mascara,alto,afinidad,limite)
    _,indices=ndimage.distance_transform_edt(~validos,return_indices=True)
    base=color/np.maximum(pesos[:,:,None],.015)
    fondo=ndimage.gaussian_filter(base[indices[0],indices[1]],(4,4,0))
    for difusion in (max(5,alto*.11),max(3,alto*.045),max(1.3,alto*.016)):
        den = ndimage.gaussian_filter(pesos,difusion)
        est = ndimage.gaussian_filter(color,(difusion,difusion,0))/np.maximum(den[:,:,None],.00001)
        mezcla = np.clip(den/.14,0,1)[:,:,None]
        fondo += mezcla*(est-fondo)
    fondo = np.stack([np.asarray(Image.fromarray(c).resize((foto.shape[1],foto.shape[0]),Image.Resampling.BILINEAR))
                      for c in fondo.transpose(2,0,1)],axis=2)
    yy, xx = np.indices(mascara.shape)
    suma = np.zeros_like(rgb)
    peso = np.zeros(mascara.shape,dtype=np.float32)
    for dy, dx in ((.86,.06),(.68,-.22),(.15,.68),(.11,-.67),(-.52,.08)):
        sy = yy + round(alto*dy)
        sx = xx + round(alto*dx)
        dentro = (sy>=0)&(sy<foto.shape[0])&(sx>=0)&(sx<foto.shape[1])
        sy, sx = sy.clip(0,foto.shape[0]-1), sx.clip(0,foto.shape[1]-1)
        valido = dentro * libre[sy,sx] * confianza[sy,sx]**2
        suma += detalle[sy,sx]*valido[:,:,None]
        peso += valido
    grano = np.clip(suma/np.maximum(peso[:,:,None],.02),-.12,.16)
    mezcla = np.clip(peso/.28,0,1)[:,:,None]
    return np.clip(fondo + mezcla*.88*grano,0,1)


def desenfoque_borde(fotograma, forma):
    """Medir la transición de luz cerca del borde físico del recipiente."""
    centro, radio, cima, pendiente = forma
    xs = centro + radio*np.linspace(-.6, .6, 31)
    margen = max(7, min(19, round(radio*.15)))
    ys = cima + pendiente*(xs-centro) + np.arange(-margen, margen+1)[:, None]
    if xs.min()<0 or xs.max()>=fotograma.width-1 or ys.min()<0 or ys.max()>=fotograma.height-1:
        return 0.
    # Leer una franja estrecha del cuadro fuente, incluso cuando el recorte
    # de la tinta sólo contiene el extremo visible del diseño.
    fila = int(ys.min())
    gris = np.asarray(fotograma.crop((0, fila, fotograma.width, math.ceil(ys.max())+1)).convert("L"), dtype=np.float32)
    perfil = ndimage.map_coordinates(gris, [ys-fila, np.broadcast_to(xs, ys.shape)], order=1)
    inicio = np.median(perfil[:3], axis=0)
    contraste = np.percentile(perfil[-4:], 75, axis=0)-inicio
    niveles = (perfil-inicio)/np.maximum(1, contraste)
    izq, medio, der = [(niveles > umbral).argmax(axis=0) for umbral in (.15, .5, .85)]
    buenos = (contraste>45) & (abs(medio-margen)<margen*.6) & (izq<=medio) & (medio<der)
    if buenos.sum()<12:
        return 0.
    # El ancho del 15 al 85 % equivale a 2,07 sigmas de un borde gaussiano.
    # Descontar la transición natural del borde y el remuestreo del video.
    sigma = np.median((der-izq)[buenos])/2.07
    return min(4.5, .88*math.sqrt(max(0., sigma*sigma-1.8**2)))


def imprimir(fotograma, caja, arte, vista, curva, estado=None, escala=1., posicion=(0.,0.), limpiar=False, textura=None):
    x, y, w, h = caja
    margen = max(6, round(max(w,h)*(.44 if estado is not None else .34)))
    mx = margen + math.ceil(w*abs(posicion[0]))
    my = margen + math.ceil(h*abs(posicion[1]))
    donante = math.ceil(h*.56) if limpiar else 0
    region = (max(0,x-mx), max(0,y-my), min(fotograma.width,x+w+mx), min(fotograma.height,y+h+my+donante))
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
    anterior = None
    px, py, normal = coordenadas_arte(u, v, w/h, vista, curva, arte.shape[1]/arte.shape[0])
    if estado is not None:
        from registro_superficie import proyectar
        px, py, borde, anterior = proyectar(estado, (xx+.5)/fx+region[0], (yy+.5)/fy+region[1], arte.shape[1]/arte.shape[0])
        # El contorno físico medido en el video distingue el fondo incluso
        # cuando la tinta blanca o negra alcanza el borde del recipiente.
        limite = borde
    mascara = (extension_tinta(foto, u, v, alto, afinidad, limite)
               if anterior is None else anterior>.35)
    if anterior is not None:
        # Completar los filetes claros y sus bordes suavizados justo al lado del trazo registrado.
        cerca = ndimage.maximum_filter(mascara,size=2*max(2,round(alto*.065))+1)
        f = foto.astype(np.float32)/255
        claro = (f[:,:,0]>.57)&(f[:,:,1]>.57*f[:,:,0])&(f[:,:,2]>.65*f[:,:,1])
        ribete = cerca & claro & (py>.09) & (py<1.03) & (limite>.55)
        mascara |= ndimage.maximum_filter(ribete,size=5)
    if limpiar and anterior is not None:
        # La máscara común puede omitir un filete que sólo se distingue durante
        # una parte del giro. Incluir trazos detectados en ese fotograma.
        extra = extension_tinta(foto,u,v,alto,afinidad,limite)
        cerca = ndimage.maximum_filter(mascara,size=max(3,round(alto*.25))|1)
        mascara |= extra & cerca & (limite>.55)
    intacto_local=None
    if limpiar and curva:
        mascara,intacto_local=superficie_limpia(foto,u,v,alto,mascara,afinidad,limite)
    opacidad = ndimage.gaussian_filter(mascara.astype(np.float32), max(.65, alto*.014))
    if intacto_local is not None:
        opacidad *= 1-.96*intacto_local
    elif np.any(afinidad<1):
        # Conservar el vidrio que ya se ve entre tintas y suavizar la transición
        # alrededor de los trazos que sí es necesario reconstruir.
        intacto = ndimage.minimum_filter(afinidad, size=max(3,round(alto*.035)))
        opacidad *= 1-.98*np.clip((intacto-.55)/.44,0,1)
    opacidad *= limite
    muestras = afinidad * np.clip((v+.025)/.08,0,1)
    excluir = mascara if intacto_local is None else mascara & (intacto_local<.86)
    if limpiar and intacto_local is not None:
        rgb=foto.astype(np.float32)/255
        saturacion=np.ptp(rgb,axis=2)/np.maximum(.03,rgb.max(2))
        material=(saturacion>.39)|(rgb.max(2)<.43)
        muestras = material*np.maximum(.38,afinidad)*np.clip((v+.17)/.11,0,1)
    fondo = (textura_de_superficie(foto, excluir, muestras, limite, alto) if limpiar else
             reconstruir(foto, mascara, alto, muestras, limite))
    if limpiar and textura is not None:
        # La misma muestra queda registrada sobre el material durante el giro;
        # usarla antes de desplazar la nueva impresión evita que ambas zonas se confundan.
        grano=ndimage.map_coordinates(textura,[py*textura.shape[0]*2.1,px*textura.shape[1]*2.1],
                                     order=1,mode="reflect")
        luz=np.mean(fondo,axis=2,keepdims=True)
        tinte=np.clip((fondo+.12)/(luz+.12),.45,1.7)
        # El relleno ya contiene detalle del fotograma. Medir cuánto admite el
        # vidrio expuesto para que la segunda muestra no duplique ese contraste.
        observado = np.mean(foto, axis=2, dtype=np.float32)/255
        libres = (muestras>.65) & (limite>.8) & ~ndimage.maximum_filter(excluir, size=7)
        if np.count_nonzero(libres) >= 32:
            residual = observado-ndimage.gaussian_filter(observado,1.75)
            real = float(np.percentile(np.abs(residual[libres]),70))
            examinar = opacidad>.5
            if np.any(examinar):
                local = luz[:,:,0]-ndimage.gaussian_filter(luz[:,:,0],1.75)
                actual = float(np.percentile(np.abs(local[examinar]),70))
                disponible = max(0.,real-actual)
                amplitud = np.clip(grano,-2*disponible,2*disponible)
                fondo=np.clip(fondo+.45*amplitud[:,:,None]*tinte,0,1)
    original_rgb = foto.astype(np.float32)/255
    base = original_rgb*(1-opacidad[:,:,None])+fondo*opacidad[:,:,None]
    # Escalar únicamente la textura en coordenadas del material, una vez
    # reconstruida la impresión antigua. Su centro permanece fijo al girar.
    # Un desplazamiento expresado como fracción del diseño frontal queda fijo
    # en la superficie incluso cuando esa superficie se ve de perfil.
    aspecto_frontal = vista[0] if curva else w/h
    proporcion = arte.shape[1]/arte.shape[0]
    px = .5 + (px-.5-posicion[0]*max(1.,aspecto_frontal/proporcion))/escala
    py = .5 + (py-.5-posicion[1]*max(1.,proporcion/aspecto_frontal))/escala
    coords = [py*(arte.shape[0]-1), px*(arte.shape[1]-1)]
    estampado = np.stack([ndimage.map_coordinates(arte[:,:,c], coords, order=1, mode="constant") for c in range(4)], 2)
    visible = (u>=0)&(u<=1)&(v>=0)&(v<=1)
    if estado is not None or escala != 1 or any(posicion):
        visible = (px>=0)&(px<=1)&(py>=0)&(py<=1)
        if curva and estado is None:
            visible &= normal > 1e-5
    densidad = .82*(.66+.34*normal)*visible*limite
    luz = np.mean(base, axis=2)
    tinta = estampado[:,:,:3]*(.73+.38*luz[:,:,None])
    alfa = estampado[:,:,3]*densidad
    suave = max(.50, min(1.25, alto/270))
    if estado is not None:
        suave = max(suave, desenfoque_borde(fotograma, estado[0])*min(fx, fy))
    alfa = ndimage.gaussian_filter(alfa, suave)
    color = ndimage.gaussian_filter(tinta*densidad[:,:,None], (suave,suave,0))
    resultado = base*(1-alfa[:,:,None]) + color
    # Solo se deben escribir los píxeles afectados, también en cuadros de tasa variable.
    activo = (opacidad+alfa) > .004
    rgba = np.dstack([np.uint8(np.clip(resultado*255+.5,0,255)), activo.astype(np.uint8)*255])
    rgba[~activo] = 0
    capa = Image.fromarray(rgba)
    return capa.resize(original,Image.Resampling.BILINEAR), region[:2]
