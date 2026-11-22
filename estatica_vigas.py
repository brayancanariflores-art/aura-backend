"""Equilibrio de vigas horizontales isostáticas con dos apoyos y cargas verticales."""
from fractions import Fraction
from math import isfinite
from soluciones import Desarrollo, Paso
from diagramas_viga import dibujar_curvas, dibujar_viga
from pydantic import BaseModel, Field


class CargaPuntual(BaseModel):
    nodo: int
    fuerza_kN_hacia_abajo: float


class ParAplicado(BaseModel):
    nodo: int
    momento_kNm_antihorario: float


class CargaDistribuida(BaseModel):
    desde_nodo: int
    hasta_nodo: int
    intensidad_inicial_kN_m_hacia_abajo: float
    intensidad_final_kN_m_hacia_abajo: float


class Viga(BaseModel):
    tramos_m: list[float] = Field(description="Distancias consecutivas entre marcas de la línea de cotas, de izquierda a derecha. El primer nodo es 0; nunca uses coordenadas acumuladas aquí.")
    apoyo_izquierdo: int = Field(description="Índice de nodo del apoyo A, no una distancia en metros.")
    apoyo_derecho: int = Field(description="Índice de nodo del apoyo B, no una distancia en metros.")
    corte: int = Field(description="Índice del nodo señalado por el punto C, no su coordenada.")
    puntuales: list[CargaPuntual]
    pares: list[ParAplicado]
    distribuidas: list[CargaDistribuida] = Field(description="Un tramo por intervalo de intensidad lineal. Una carga triangular con vértice interior requiere dos tramos, uno creciente y uno decreciente.")
    cortante_positivo_hacia_abajo: bool = Field(description="En la cara del tramo izquierdo; usa verdadero salvo convención contraria indicada.")
    momento_positivo_antihorario: bool = Field(description="En la cara del tramo izquierdo; usa verdadero salvo convención contraria indicada.")


def resultante_lineal(x0, x1, w0, w1, limite):
    """Carga y primer momento respecto al origen, hasta el límite."""
    longitud = max(0, min(limite, x1) - x0)
    pendiente = (w1 - w0) / (x1 - x0)
    fuerza = w0 * longitud + pendiente * longitud**2 / 2
    momento = x0 * fuerza + w0 * longitud**2 / 2 + pendiente * longitud**3 / 3
    return fuerza, momento


def medida(valor):
    if not isfinite(valor):
        raise ValueError("medida no finita")
    return Fraction(str(valor))


def desarrollar_viga(viga, diagramas=None):
    x = [Fraction(0)]
    for tramo in viga.tramos_m:
        if not isfinite(tramo) or tramo <= 0:
            raise ValueError("cotas")
        x.append(x[-1] + medida(tramo))
    def posicion(indice):
        if not 0 <= indice < len(x):
            raise ValueError("posición")
        return x[indice]
    a, b, c = map(posicion, (viga.apoyo_izquierdo, viga.apoyo_derecho, viga.corte))
    if not a < b or not 0 < c < x[-1] or c in (a, b):
        raise ValueError("posición de apoyos o corte")

    fuerzas = [(posicion(p.nodo), medida(p.fuerza_kN_hacia_abajo)) for p in viga.puntuales]
    pares = [(posicion(p.nodo), medida(p.momento_kNm_antihorario)) for p in viga.pares]
    if any(pos == c for pos, _ in fuerzas + pares):
        raise ValueError("lado del corte en la discontinuidad")
    tramos = []
    for q in viga.distribuidas:
        inicio, fin = posicion(q.desde_nodo), posicion(q.hasta_nodo)
        if inicio >= fin:
            raise ValueError("tramo de carga")
        tramos.append((inicio, fin, medida(q.intensidad_inicial_kN_m_hacia_abajo), medida(q.intensidad_final_kN_m_hacia_abajo)))

    def cargas(limite, incluir_nodo=True):
        fuerza = sum(p for pos, p in fuerzas if pos < limite or (incluir_nodo and pos == limite))
        momento = sum(pos * p for pos, p in fuerzas if pos < limite or (incluir_nodo and pos == limite))
        for tramo in tramos:
            f, m = resultante_lineal(*tramo, limite)
            fuerza += f
            momento += m
        return fuerza, momento, sum(m for pos, m in pares if pos < limite or (incluir_nodo and pos == limite))

    f, m, par = cargas(x[-1])
    rb = (m - a*f - par) / (b-a)
    ra = f - rb
    fc, mc, pc = cargas(c)
    reacciones = [(a, ra), (b, rb)]
    def esfuerzos(limite, incluir_nodo):
        carga, primer_momento, par_local = cargas(limite, incluir_nodo)
        apoyos = [(pos, r) for pos, r in reacciones if pos < limite or (pos == limite and incluir_nodo)]
        v = sum(r for pos, r in apoyos) - carga
        m_local = sum((limite-pos)*r for pos, r in apoyos) - limite*carga + primer_momento - par_local
        return v, m_local
    cortante, flector = esfuerzos(c, True)
    derecha_v = (f-fc) - sum(r for pos, r in reacciones if pos > c)
    derecha_m = sum((pos-c)*r for pos, r in reacciones if pos > c) - ((m-mc)-c*(f-fc)) + par-pc
    if cortante != derecha_v or flector != derecha_m:
        raise ValueError("equilibrio")
    if diagramas is not None and len(x) <= 65 and len(tramos)+len(fuerzas)+len(pares) <= 128:
        def seccion(pos, incluir_nodo):
            v, momento = esfuerzos(pos, incluir_nodo)
            return (v if viga.cortante_positivo_hacia_abajo else -v,
                    momento if viga.momento_positivo_antihorario else -momento)
        diagramas["cuerpo"] = dibujar_viga(x, c, reacciones, fuerzas, pares, tramos, cortante, flector)
        diagramas["curvas"] = dibujar_curvas(x, c, seccion)
    if not viga.cortante_positivo_hacia_abajo:
        cortante = -cortante
    if not viga.momento_positivo_antihorario:
        flector = -flector
    def numero(valor, tex=False):
        valor = Fraction(valor)
        # Un decimal corto puede escribirse exactamente; en los demás casos conservamos la fracción.
        for decimales in range(7):
            escala = 10**decimales
            if escala % valor.denominator == 0:
                entero, resto = divmod(abs(valor.numerator) * escala // valor.denominator, escala)
                signo = "-" if valor < 0 else ""
                coma = "{,}" if tex else ","
                return signo + str(entero) + (f"{coma}{resto:0{decimales}d}" if decimales else "")
        if tex:
            signo = "-" if valor < 0 else ""
            return rf"{signo}\frac{{{abs(valor.numerator)}}}{{{valor.denominator}}}"
        return str(valor)

    def t(valor):
        return numero(valor, tex=True)

    sentido_v = "abajo" if viga.cortante_positivo_hacia_abajo else "arriba"
    sentido_m = "antihorario" if viga.momento_positivo_antihorario else "horario"
    exacto = rf"N_C=0\,\mathrm{{kN}},\quad V_C={t(cortante)}\,\mathrm{{kN}},\quad M_C={t(flector)}\,\mathrm{{kN{{\cdot}}m}}"
    lectura = f"Respuesta en C: N = 0 kN, V = {numero(cortante)} kN, M = {numero(flector)} kN·m."
    decimales = []
    for simbolo, valor, unidad in (("V_C", cortante, "kN"), ("M_C", flector, r"kN{\cdot}m")):
        if "/" in numero(valor):
            decimal = format(float(valor), ".6g").replace(".", "{,}")
            decimales.append(rf"{simbolo}\approx {decimal}\,\mathrm{{{unidad}}}")
    nota = f"En la cara izquierda, V positivo apunta hacia {sentido_v} y M positivo es {sentido_m}."
    pasos = [
        Paso(explicacion="Mide desde el extremo izquierdo para ubicar los dos apoyos y el punto C.",
             formula=rf"x_A={t(a)}\,\mathrm{{m}},\quad x_B={t(b)}\,\mathrm{{m}},\quad x_C={t(c)}\,\mathrm{{m}}"),
        Paso(explicacion="Al tomar momentos alrededor de A, la fuerza del propio apoyo queda fuera de la cuenta porque actúa justo allí. Así encontramos la reacción en B y luego la de A.",
             formula=rf"W={t(f)}\,\mathrm{{kN}},\quad R_B=\frac{{{t(m-a*f-par)}}}{{{t(b-a)}}}={t(rb)}\,\mathrm{{kN}},\quad R_A=W-R_B={t(ra)}\,\mathrm{{kN}}"),
        Paso(explicacion="Imagina que cortas la viga en C. Al equilibrar el tramo izquierdo, encuentras las fuerzas y el momento que transmite el corte.",
             formula=rf"W_{{\mathrm{{izq}}}}={t(fc)}\,\mathrm{{kN}},\quad \sum F_x=0,\quad \sum F_y=0,\quad \sum M_C=0")
    ]
    return Desarrollo(pasos=pasos, exacto=exacto, aproximacion=r",\quad ".join(decimales), nota=nota, lectura=lectura)


def resolver_viga(viga):
    return desarrollar_viga(viga).lectura
