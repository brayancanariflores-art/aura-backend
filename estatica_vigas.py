"""Equilibrio de vigas horizontales isostáticas con dos apoyos y cargas verticales."""
from math import isclose, isfinite
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


def resolver_viga(viga):
    x = [0.0]
    for tramo in viga.tramos_m:
        if not isfinite(tramo) or tramo <= 0:
            raise ValueError("cotas")
        x.append(x[-1] + tramo)
    def posicion(indice):
        if not 0 <= indice < len(x):
            raise ValueError("posición")
        return x[indice]
    a, b, c = map(posicion, (viga.apoyo_izquierdo, viga.apoyo_derecho, viga.corte))
    if not a < b or not 0 < c < x[-1] or c in (a, b):
        raise ValueError("posición de apoyos o corte")

    fuerzas = [(posicion(p.nodo), p.fuerza_kN_hacia_abajo) for p in viga.puntuales]
    pares = [(posicion(p.nodo), p.momento_kNm_antihorario) for p in viga.pares]
    if any(pos == c for pos, _ in fuerzas + pares):
        raise ValueError("lado del corte en la discontinuidad")
    tramos = []
    for q in viga.distribuidas:
        inicio, fin = posicion(q.desde_nodo), posicion(q.hasta_nodo)
        if inicio >= fin:
            raise ValueError("tramo de carga")
        tramos.append((inicio, fin, q.intensidad_inicial_kN_m_hacia_abajo, q.intensidad_final_kN_m_hacia_abajo))

    def cargas(limite):
        fuerza = sum(p for pos, p in fuerzas if pos <= limite)
        momento = sum(pos * p for pos, p in fuerzas if pos <= limite)
        for tramo in tramos:
            f, m = resultante_lineal(*tramo, limite)
            fuerza += f
            momento += m
        return fuerza, momento, sum(m for pos, m in pares if pos <= limite)

    f, m, par = cargas(x[-1])
    rb = (m - a*f - par) / (b-a)
    ra = f - rb
    fc, mc, pc = cargas(c)
    reacciones = [(a, ra), (b, rb)]
    cortante = sum(r for pos, r in reacciones if pos < c) - fc
    flector = sum((c-pos)*r for pos, r in reacciones if pos < c) - (c*fc-mc) - pc
    derecha_v = (f-fc) - sum(r for pos, r in reacciones if pos > c)
    derecha_m = sum((pos-c)*r for pos, r in reacciones if pos > c) - ((m-mc)-c*(f-fc)) + par-pc
    if not (isclose(cortante, derecha_v, abs_tol=1e-8) and isclose(flector, derecha_m, abs_tol=1e-8)):
        raise ValueError("equilibrio")
    if not viga.cortante_positivo_hacia_abajo:
        cortante = -cortante
    if not viga.momento_positivo_antihorario:
        flector = -flector
    def n(valor):
        return format(0 if abs(valor) < 1e-9 else valor, ".8g").replace(".", ",")
    sentido_v = "abajo" if viga.cortante_positivo_hacia_abajo else "arriba"
    sentido_m = "antihorario" if viga.momento_positivo_antihorario else "horario"
    return (f"Respuesta en C: N = 0 kN, V = {n(cortante)} kN, M = {n(flector)} kN·m.\n"
            f"Distancias desde el extremo izquierdo: A = {n(a)} m, B = {n(b)} m, C = {n(c)} m.\n"
            f"Reacciones verticales: A = {n(ra)} kN, B = {n(rb)} kN (positivo hacia arriba). "
            f"Carga vertical total = {n(f)} kN; hasta C = {n(fc)} kN.\n"
            f"Convención en la cara del tramo izquierdo: V positivo hacia {sentido_v}, M positivo {sentido_m}.")
