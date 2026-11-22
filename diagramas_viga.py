"""Dibujos vectoriales de la geometría y el equilibrio calculados por AURA."""
from math import atan2, cos, sin


def numero(v):
    return format(float(v), ".6g").replace(".", ",")


def linea(x, y, u, v, color="#496174", ancho=1.5):
    return f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{u:.2f}" y2="{v:.2f}" stroke="{color}" stroke-width="{ancho}"/>'


def texto(x, y, s, ancla="middle", color="#163247"):
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{ancla}" fill="{color}">{s}</text>'


def flecha(x, y, u, v, color):
    ang = atan2(v-y, u-x)
    alas = [(u-6*cos(ang+t), v-6*sin(ang+t)) for t in (-.45, .45)]
    return linea(x, y, u, v, color, 2) + ''.join(linea(u, v, a, b, color, 2) for a, b in alas)


def giro(x, y, antihorario):
    azul = "#0c6597"
    arriba, abajo = (y+14, y-14) if antihorario else (y-14, y+14)
    barrido = 0 if antihorario else 1
    return (f'<path d="M{x},{arriba} A14,14 0 0,{barrido} {x},{abajo}" fill="none" stroke="{azul}" stroke-width="2"/>'
            + flecha(x+5, abajo+(1 if antihorario else -1), x-1, abajo, azul))


def figura(alto, contenido):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 {alto}" '
            'font-family="sans-serif" font-size="14"><rect width="100%" height="100%" fill="#fff"/>'
            + ''.join(contenido) + '</svg>')


def dibujar_viga(nodos, c, reacciones, fuerzas, pares, cargas, cortante, momento):
    piezas = []
    escala_cargas = max((max(abs(w0), abs(w1)) for _, _, w0, w1 in cargas), default=1) or 1
    for limite, y, titulo in ((nodos[-1], 92, "Viga completa"), (c, 304, "Tramo izquierdo hasta C")):
        corte = limite == c
        def px(pos):
            return 47 + 345 * float(pos / limite)

        piezas.append(texto(12, y-72, titulo, "start"))
        piezas.append(linea(px(0), y, px(limite), y, "#324e60", 8))
        for ini, fin, w0, w1 in cargas:
            if ini >= limite:
                continue
            extremo = min(fin, limite)
            wf = w0 + (w1-w0)*(extremo-ini)/(fin-ini)
            def altura(w):
                return y-6-34*float(abs(w)/escala_cargas)
            pasos = max(2, int((px(extremo)-px(ini))/17))
            for i in range(pasos+1):
                w = w0 + (wf-w0)*i/pasos
                if w:
                    x = px(ini)+(px(extremo)-px(ini))*i/pasos
                    a, b = (altura(w), y-6) if w > 0 else (y-6, altura(w))
                    piezas.append(flecha(x, a, x, b, "#176e85"))
            cruces = [(ini, w0), (extremo, wf)]
            if w0 * wf < 0:
                cruces.insert(1, (ini + (extremo-ini)*w0/(w0-wf), 0))
            for (x1, v1), (x2, v2) in zip(cruces, cruces[1:]):
                piezas.append(linea(px(x1), altura(v1), px(x2), altura(v2), "#176e85"))
            leyenda = numero(w0) if wf == w0 else f"{numero(w0)} → {numero(wf)}"
            piezas.append(texto((px(ini)+px(extremo))/2, y-49, leyenda+" kN/m"))
        for pos, p in fuerzas:
            if pos <= limite and p:
                x = px(pos)
                desde, hasta = (y-45,y-6) if p > 0 else (y-6,y-45)
                piezas.extend([flecha(x, desde, x, hasta, "#176e85"), texto(x,y-50,numero(abs(p))+" kN")])
        for pos, par in pares:
            if pos <= limite and par:
                x = px(pos)
                piezas.extend([giro(x,y-23,par>0),texto(x,y-47,numero(abs(par))+" kN·m")])
        for nombre, (pos, r) in zip(("A","B"), reacciones):
            if pos > limite:
                continue
            x = px(pos)
            piezas.append(f'<path d="M{x},{y+4} l-7,11 h14 Z" fill="#8295a2"/>')
            if nombre == "B":
                piezas.append(linea(x-8,y+18,x+8,y+18))
            if r:
                desde,hasta = (y+44,y+20) if r>0 else (y+20,y+44)
                piezas.append(flecha(x,desde,x,hasta,"#ac4b20"))
            piezas.append(texto(x,y+60,f"{nombre}: {numero(abs(r))} kN"))
        xc = px(c)
        piezas.extend([linea(xc,y-7,xc,y+8,"#9b344d",2), texto(xc-4,y+22,"C","end")])
        for p, q in zip(nodos, nodos[1:]):
            if q > limite:
                break
            lo, hi = px(p), px(q)
            piezas.extend([linea(lo,y+77,hi,y+77), linea(lo,y+73,lo,y+81),
                           linea(hi,y+73,hi,y+81), texto((lo+hi)/2,y+96,numero(q-p)+" m")])
        if corte:
            if cortante:
                piezas.append(flecha(xc+8,y,xc+8,y+(30 if cortante>0 else -30),"#9b344d"))
                piezas.append(texto(xc+14, y+(30 if cortante>0 else -25), "V", "start"))
            if momento:
                piezas.extend([giro(xc+31,y,momento>0), texto(xc+51,y+4,"M","start")])
            piezas.append(texto(12,y+122,f"N = 0 kN    ·    |V| = {numero(abs(cortante))} kN    ·    |M| = {numero(abs(momento))} kN·m","start"))
    return figura(444,piezas)


def dibujar_curvas(nodos, c, seccion):
    # Se evalúan ambos lados de cada nodo para dibujar los saltos por cargas y pares.
    muestras = [(0., *map(float, seccion(nodos[0], False)))]
    for ini, fin in zip(nodos, nodos[1:]):
        for i in range(25):
            x = ini + (fin-ini)*i/24
            muestras.append((float(x), *map(float,seccion(x, i != 24))))
    muestras.append((float(nodos[-1]), *map(float, seccion(nodos[-1], True))))
    piezas = []
    for indice, unidad, titulo, techo in ((1,"kN","Fuerza cortante V",0),(2,"kN·m","Momento flector M",162)):
        valores = [fila[indice] for fila in muestras]
        minimo, maximo = min(0,*valores), max(0,*valores)
        escala = maximo-minimo or 1
        def px(x):
            return 70+341*float(x/nodos[-1])
        def py(v):
            return techo+112-72*(v-minimo)/escala
        puntos = " ".join(f"{px(fila[0]):.2f},{py(fila[indice]):.2f}" for fila in muestras)
        piezas.extend([texto(12,techo+20,titulo+" ("+unidad+")","start"),linea(70,py(0),415,py(0)),
                       texto(64,py(0)+4,"0","end"),linea(px(c),techo+32,px(c),techo+118,"#c4cdd3"),
                       f'<polyline points="{puntos}" fill="none" stroke="#166f8b" stroke-width="2.5"/>'])
        for extremo in (minimo, maximo):
            if extremo and abs(py(extremo) - py(0)) > 14:
                piezas.append(texto(64, py(extremo)+4, format(extremo, ".5g").replace(".", ","), "end"))
        vc = float(seccion(c,True)[indice-1])
        piezas.extend([f'<circle cx="{px(c):.2f}" cy="{py(vc):.2f}" r="4" fill="#ac4b20"/>',
                       texto(70,techo+136,"0 m","start"),texto(px(c),techo+136,"C"),texto(412,techo+136,numero(nodos[-1])+" m","end"),
                       texto(12,techo+157,f"En C: {numero(vc)} {unidad}","start")])
    return figura(328,piezas)
