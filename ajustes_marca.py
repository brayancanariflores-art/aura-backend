"""Instrucciones de composición en la superficie del objeto, independientes del borrado."""
import io
import json
import math
import re

from PIL import Image

from estudio_medios import ejecutar, entrada_video
from tamano_marca import normalizar


class AjusteNoAplicado(ValueError):
    """Un diagnóstico de edición que se puede mostrar en el chat."""


def pide_posicion(texto):
    return bool(re.search(r"\b(centr(?:a|ar|alo|ala|ado|ada)|al (?:centro|medio)|en el (?:centro|medio)|"
                          r"(?:mas|hacia) (?:abajo|arriba|la izquierda|la derecha)|"
                          r"(?:baj|sub|muev|desplaz|reubic|reposicion)\w*\b.{0,40}\b(?:logo|marca|sello|pixeles|px)\b)",
                          normalizar(texto)))


def pide_limpieza(texto):
    return bool(re.search(r"\b(?:mancha|borros\w*|textura|pinza|restos?|rayas|limpi\w*|reconstru\w*)\b",
                          normalizar(texto)))


def desplazamiento_explicito(texto):
    """Desplazamientos expresados en píxeles del video, sean una o dos direcciones."""
    direcciones = {'baj': (0, 1), 'abajo': (0, 1), 'sub': (0, -1), 'arriba': (0, -1),
                   'izquierda': (-1, 0), 'derecha': (1, 0)}
    dx = dy = 0.
    encontrado = False
    for parte in re.split(r'\s+y\s+|[.;]', normalizar(texto)):
        numero = re.search(r'\b(\d+(?:[.,]\d+)?)\s*(?:px|pixel(?:es)?)\b', parte)
        direccion = re.search(r'\b(abajo|arriba|izquierda|derecha)\b', parte)
        if not direccion:
            direccion = re.search(r'\b(baj|sub)\w*', parte)
        if numero and direccion:
            valor = float(numero[1].replace(',', '.'))
            x, y = direcciones[direccion[1]]
            dx += x * valor
            dy += y * valor
            encontrado = True
    return (dx, dy) if encontrado else None


def ancla_frontal(grupos):
    puntos = [p for grupo, _ in grupos for p in grupo]
    anchos = sorted(p[3] for p in puntos)
    candidatos = [p for p in puntos if p[3] >= anchos[int(.7*(len(anchos)-1))]]
    return candidatos[len(candidatos)//2]


def planificar(clip, grupos, tam, pedido, previo, cliente, modelo):
    x, y = previo.get('posicion', (0., 0.))
    limpiar = previo.get('limpiar', False) or pide_limpieza(pedido)
    if pide_posicion(pedido):
        t, izq, sup, w, h = ancla_frontal(grupos)
        delta = desplazamiento_explicito(pedido)
        if delta is not None:
            x += delta[0]/w
            y += delta[1]/h
        else:
            from seguimiento_marca import imagen_chat
            # Imagen entera: el centro deseado se mide respecto al objeto, no al cuadro de la marca.
            jpg = ejecutar(['-v','error','-ss',str(t),*entrada_video(clip),'-frames:v','1',
                            '-vf',"scale='min(640,iw)':-2",'-c:v','mjpeg','-f','image2pipe','pipe:1'],salida=True)
            with Image.open(io.BytesIO(jpg)) as im:
                imagen = imagen_chat(im)
            actual = [(izq+w*(.5+x))/tam[0], (sup+h*(.5+y))/tam[1]]
            guia = ('Eres el compositor de un reel. La imagen muestra la toma original. '
                    'Indica el centro deseado de la NUEVA impresión, siguiendo el encargo y los bordes físicos '
                    'de la cara útil del objeto. Respeta las direcciones y los ejes no solicitados. '
                    'Las coordenadas son fracciones del cuadro completo, origen arriba a la izquierda. '
                    'Si no distingues el objeto responde {"visible":false}. En otro caso devuelve JSON '
                    '{"visible":true,"x":0.5,"y":0.5} con las coordenadas medidas.')
            respuesta = cliente.with_options(timeout=45,max_retries=1).chat.completions.create(
                model=modelo,response_format={'type':'json_object'},
                messages=[{'role':'system','content':guia},{'role':'user','content':[
                    {'type':'text','text':f'Centro actual (x,y): {actual}. Pedido: {pedido}'},imagen]}])
            datos=json.loads(respuesta.choices[0].message.content or '{}')
            xy=[datos.get(k) for k in ('x','y')]
            if datos.get('visible') is not True or not all(type(v) in (float,int) and 0<v<1 for v in xy):
                raise AjusteNoAplicado('No pude medir el centro del objeto. Indica cuántos píxeles debo mover el logo.')
            x=(xy[0]*tam[0]-izq)/w-.5
            y=(xy[1]*tam[1]-sup)/h-.5
    if not all(math.isfinite(v) and abs(v)<=1.75 for v in (x,y)):
        raise AjusteNoAplicado('El desplazamiento solicitado sale de la zona de impresión que puedo seguir.')
    return {'posicion':[x,y],'limpiar':limpiar}
