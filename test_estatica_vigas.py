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

    def test_dibujo_pedido_como_seguimiento_en_el_chat(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from diagrama_chat import PedidoViga, pide_dibujo_viga, responder_con_diagrama
        historial = [{"role": "user", "content": "Una viga de 4 m con dos apoyos y 10 kN en el centro."},
                     {"role": "assistant", "content": "Cada apoyo sostiene 5 kN."}]
        pregunta = "¿Me das el diagrama?"
        self.assertTrue(pide_dibujo_viga("me das el diagrama", historial))
        self.assertTrue(pide_dibujo_viga("y el diagrama de esta ultima", historial))
        self.assertFalse(pide_dibujo_viga("busca en internet el ultimo diagrama de vigas", historial))
        self.assertFalse(pide_dibujo_viga("cuanto sostiene cada apoyo", historial))
        cliente = Mock()
        pedido = PedidoViga(corresponde=True, viga=viga([2, 2], 0, 2, None, puntuales=[(1, 10)]),
                            curvas=False, aclaracion="")
        cliente.responses.parse.return_value = SimpleNamespace(output_parsed=pedido)
        salida = responder_con_diagrama(cliente, "modelo", historial, pregunta)
        self.assertEqual(cliente.responses.parse.call_args.kwargs["input"], historial + [{"role": "user", "content": pregunta}])
        item, = salida["ejercicios"]
        self.assertTrue(item["soloDiagramas"])
        self.assertEqual(set(item["diagramas"]), {"cuerpo"})
        etiquetas = list(ElementTree.fromstring(item["diagramas"]["cuerpo"]).itertext())
        self.assertIn("10 kN", etiquetas)
        self.assertIn("A: 5 kN", etiquetas)
        self.assertIn("B: 5 kN", etiquetas)
        self.assertEqual(etiquetas.count("2 m"), 2)
        self.assertNotIn("C", etiquetas)
        pedido.viga = None
        pedido.aclaracion = "¿A qué distancia está la carga?"
        self.assertEqual(responder_con_diagrama(cliente, "modelo", historial, pregunta), {"respuesta": pedido.aclaracion})
        pedido.corresponde = False
        self.assertIsNone(responder_con_diagrama(cliente, "modelo", historial, pregunta))

    def test_viga_completa_con_carga_fuera_del_centro(self):
        datos = viga([1, 3], 0, 2, None, puntuales=[(1, 10)])
        diagramas = {}
        respuesta = desarrollar_viga(datos, diagramas)
        self.assertIn("A = 7,5 kN", respuesta.lectura)
        self.assertIn("B = 2,5 kN", respuesta.lectura)
        etiquetas = list(ElementTree.fromstring(diagramas["curvas"]).itertext())
        self.assertIn("Momento flector M (kN·m)", etiquetas)
        self.assertNotIn("C", etiquetas)

    def test_conserva_fracciones_y_aproxima_debajo(self):
        datos = viga([1,1,1], 0,3,1, puntuales=[(2,1)])
        solucion = desarrollar_viga(datos)
        self.assertIn(r"\frac{1}{3}", solucion.exacto)
        self.assertIn(r"\approx 0{,}333333", solucion.aproximacion)
        self.assertTrue(all(paso.explicacion and paso.formula for paso in solucion.pasos))

    def test_se_requiere_cara_si_hay_carga_en_el_corte(self):
        with self.assertRaises(ValueError):
            resolver_viga(viga([2,3], 0,2,1, puntuales=[(1,12)]))
