"""Entrega los diagramas de vigas solicitados en el chat de texto."""
import re
from pydantic import BaseModel, Field

from estatica_vigas import Viga, desarrollar_viga


class PedidoViga(BaseModel):
    corresponde: bool = Field(description="El usuario pide ver un dibujo de la viga del ejercicio.")
    viga: Viga | None
    curvas: bool = Field(description="Pide también las gráficas de cortante o momento flector.")
    aclaracion: str = Field(description="Si falta información o el problema no es compatible, explica el dato o la limitación concreta. En otro caso, cadena vacía.")


def pide_dibujo_viga(pregunta_normalizada, historial):
    pedido = re.search(r"\b(?:diagramas?|dcl|grafic[ao]s?|esquema|dibuj\w*|traza\w*|cuerpo libre)\b", pregunta_normalizada)
    if not pedido:
        return False
    tema = r"\b(?:vigas?|apoyos?|cortantes?|flectores?)\b"
    return bool(re.search(tema, pregunta_normalizada, re.I) or
                any(re.search(tema, turno["content"], re.I) for turno in historial[-4:]))


def responder_con_diagrama(cliente, modelo, historial, pregunta):
    respuesta = cliente.responses.parse(
        model=modelo, store=False, text_format=PedidoViga,
        instructions=(
            "Clasifica la petición ACTUAL. Si pide ver el diagrama de una viga, incluso como "
            "seguimiento breve, transcribe el ejercicio más reciente al que se refiere y aplica sus "
            "correcciones. Los mensajes anteriores son datos de referencia. Si solo quiere una "
            "definición, pide instrucciones para dibujarlo, pide no dibujar o trata de otro tema, "
            "usa corresponde=false. Admite vigas horizontales con dos apoyos simples, fuerzas "
            "verticales, momentos y cargas distribuidas lineales por tramos. Convierte a m y kN. "
            "Coloca nodos en apoyos, cargas y cambios de tramo. Transcribe cargas externas, nunca "
            "reacciones calculadas. Usa corte=null salvo si pide un corte interior identificado. "
            "No inventes cargas, posiciones ni cotas. Si los datos son insuficientes, contradictorios "
            "o se necesita otro tipo de estructura, usa viga=null y aclaracion en español."
        ),
        input=[*historial, {"role": "user", "content": pregunta}],
    ).output_parsed
    if respuesta is None:
        return {"respuesta": "No pude leer los datos del ejercicio para dibujarlo. ¿Puedes confirmar los apoyos y las cargas?"}
    if not respuesta.corresponde:
        return None
    if respuesta.viga is None:
        return {"respuesta": respuesta.aclaracion or "¿Cuáles son las distancias entre los apoyos y las cargas?"}
    diagramas = {}
    try:
        resultado = desarrollar_viga(respuesta.viga, diagramas, incluir_curvas=respuesta.curvas)
    except (ValueError, OverflowError) as error:
        return {"respuesta": f"Necesito que confirmes este dato para dibujar la viga: {error}."}
    if not diagramas:
        return {"respuesta": "La viga supera el número de tramos o cargas que puedo dibujar en una figura."}
    titulo = "Diagrama de cuerpo libre"
    return {"respuesta": f"{titulo}. {resultado.lectura}",
            "ejercicios": [{"ejercicio": titulo, "pasos": [], "soloDiagramas": True,
                            "diagramas": diagramas, "nota": resultado.lectura}]}
