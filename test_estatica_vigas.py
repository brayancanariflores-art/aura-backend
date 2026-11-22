import unittest
from xml.etree import ElementTree
from estatica_vigas import CargaDistribuida, CargaPuntual, ParAplicado, Viga, desarrollar_viga, resolver_viga


def viga(tramos, a, b, c, distribuidas=(), puntuales=(), pares=()):
    return Viga(tramos_m=tramos, apoyo_izquierdo=a, apoyo_derecho=b, corte=c,
                distribuidas=[CargaDistribuida(desde_nodo=i, hasta_nodo=j,
                    intensidad_inicial_kN_m_hacia_abajo=u, intensidad_final_kN_m_hacia_abajo=v)
                    for i, j, u, v in distribuidas],
                puntuales=[CargaPuntual(nodo=i, fuerza_kN_hacia_abajo=f) for i,f in puntuales],
                pares=[ParAplicado(nodo=i, momento_kNm_antihorario=m) for i,m in pares],
                momento_positivo_antihorario=True, cortante_positivo_hacia_abajo=True)


class Equilibrio(unittest.TestCase):
    def test_corte_entre_dos_voladizos(self):
        resultado = resolver_viga(viga([2,1,1,2], 1,3,2, puntuales=[(4,10)], pares=[(0,20)]))
        self.assertIn("V = 0 kN, M = -20 kN·m", resultado)

    def test_carga_escalonada_y_convencion(self):
        datos = viga([1.5,1.5], 0,2,1, [(0,1,4,4),(1,2,8,8)])
        self.assertIn("V = 1,5 kN, M = 6,75 kN·m", resolver_viga(datos))
        datos.cortante_positivo_hacia_abajo = False
        datos.momento_positivo_antihorario = False
        self.assertIn("V = -1,5 kN, M = -6,75 kN·m", resolver_viga(datos))

    def test_corte_en_intermedio_de_una_carga_triangular(self):
        datos = viga([1.5,1.5,3], 0,3,1, [(0,2,0,6),(2,3,6,0)])
        self.assertIn("V = 6,75 kN, M = 12,375 kN·m", resolver_viga(datos))

    def test_diagramas_reflejan_el_corte_y_la_convencion(self):
        datos = viga([1.5,1.5], 0,2,1, [(0,1,4,4),(1,2,8,8)])
        datos.momento_positivo_antihorario = False
        diagramas = {}
        solucion = desarrollar_viga(datos, diagramas)
        esquema = list(ElementTree.fromstring(diagramas["cuerpo"]).itertext())
        curvas = list(ElementTree.fromstring(diagramas["curvas"]).itertext())
        self.assertIn("Tramo izquierdo hasta C", esquema)
        self.assertIn("B: 10,5 kN", esquema)
        self.assertIn("En C: 1,5 kN", curvas)
        self.assertIn("En C: -6,75 kN·m", curvas)
        self.assertIn("M = -6,75 kN·m", solucion.lectura)

    def test_conserva_fracciones_y_aproxima_debajo(self):
        datos = viga([1,1,1], 0,3,1, puntuales=[(2,1)])
        solucion = desarrollar_viga(datos)
        self.assertIn(r"\frac{1}{3}", solucion.exacto)
        self.assertIn(r"\approx 0{,}333333", solucion.aproximacion)
        self.assertTrue(all(paso.explicacion and paso.formula for paso in solucion.pasos))

    def test_se_requiere_cara_si_hay_carga_en_el_corte(self):
        with self.assertRaises(ValueError):
            resolver_viga(viga([2,3], 0,2,1, puntuales=[(1,12)]))
