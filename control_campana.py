"""Controles sobre la toma sin rótulos y sobre la identidad del anuncio exportado."""
import json

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares

from ajustes_marca import AjusteNoAplicado
from estudio_medios import entrada_video, ejecutar, inspeccionar
from seguimiento_marca import imagen_chat


def contiene_accion(anterior, actual):
    """Compensar paneo, zoom y luminosidad antes de buscar movimiento dentro del plano."""
    alto, ancho = anterior.shape
    y, x = np.mgrid[5:alto-5:2, 5:ancho-5:2].astype("float32")
    centro_x, centro_y = (ancho - 1) / 2, (alto - 1) / 2
    objetivo = actual[5:alto-5:2, 5:ancho-5:2].astype("float32")
    origen = anterior.astype("float32")

    def diferencia(parametros):
        zoom, dx, dy, luz = parametros
        coords = ((y - centro_y) * zoom + centro_y + dy, (x - centro_x) * zoom + centro_x + dx)
        estimada = ndimage.map_coordinates(origen, coords, order=1, mode="nearest")
        return (estimada + luz - objetivo).ravel().astype(float)

    def jacobiano(parametros):
        inicial = diferencia(parametros)
        # La interpolación sobre float32 necesita pasos finitos en unidades visibles.
        pasos = (.002, .12, .12, .20)
        columnas = []
        for i, paso in enumerate(pasos):
            nuevo = parametros.copy()
            nuevo[i] += paso
            columnas.append((diferencia(nuevo) - inicial) / paso)
        return np.column_stack(columnas)

    # Inicializar con la correlación de fase; un borde fino puede moverse más de un
    # píxel entre muestras y quedar fuera del alcance de un ajuste por gradiente.
    fase = np.fft.rfft2(origen) * np.conj(np.fft.rfft2(actual))
    correlacion = np.fft.irfft2(fase / np.maximum(np.abs(fase), 1e-8), s=origen.shape, axes=(0, 1))
    dy, dx = np.unravel_index(np.argmax(correlacion), correlacion.shape)
    dx = (int(dx) + ancho // 2) % ancho - ancho // 2
    dy = (int(dy) + alto // 2) % alto - alto // 2
    inicial = (1., 0., 0., 0.)
    if abs(dx) <= 12 and abs(dy) <= 12:
        desplazado = (1., float(dx), float(dy), 0.)
        if np.median(abs(diferencia(desplazado))) < np.median(abs(diferencia(inicial))):
            inicial = desplazado
    ajuste = least_squares(diferencia, inicial, jac=jacobiano,
                           bounds=((.77, -12., -12., -35.), (1.28, 12., 12., 35.)),
                           loss="soft_l1", f_scale=5, max_nfev=15)
    residual = np.abs(ajuste.fun)
    # Un corte puede cambiar todo el encuadre. Por sí solo no aporta pruebas de acción.
    if np.median(residual) > 27:
        return False
    return bool(np.mean(residual > 9) > .085)


def revisar_accion(ruta):
    ancho, alto = 72, 80
    datos = ejecutar(["-v", "error", *entrada_video(ruta), "-an", "-vf", f"fps=4,scale={ancho}:{alto},format=gray",
                      "-pix_fmt", "gray", "-frames:v", "64", "-f", "rawvideo", "pipe:1"], salida=True, tiempo=30)
    cuadros = np.frombuffer(datos, np.uint8).reshape(-1, alto, ancho)
    if len(cuadros) < 6:
        raise AjusteNoAplicado("La toma es demasiado corta para revisar la acción.")
    analisis = [contiene_accion(a, b) for a, b in zip(cuadros[:-1:2], cuadros[1::2])]
    if sum(analisis) < max(2, len(analisis) * .23):
        raise AjusteNoAplicado("En la toma solo pude confirmar fotos, cortes o movimiento de cámara. Falta verificar acción dentro del plano.")


def imagen_de_marca(ruta):
    with Image.open(ruta) as im:
        arte = im.convert("RGBA")
        arte.thumbnail((512, 512))
        vista = Image.new("RGB", arte.size, "white")
        vista.paste(arte, mask=arte.getchannel("A"))
    return imagen_chat(vista)


def revisar_identidad(ruta, logo, pedido, cliente, modelo):
    if cliente is None:
        raise AjusteNoAplicado("Falta comprobar el resultado visual de la campaña.")
    _, muestras = inspeccionar(ruta)
    if len(muestras) < 4:
        raise AjusteNoAplicado("No pude comprobar la secuencia y el cierre del anuncio.")
    partes = [{"type": "text", "text": "Pedido del cliente: " + pedido}]
    if logo:
        partes.extend([{"type": "text", "text": "Arte oficial de referencia:"}, imagen_de_marca(logo)])
    partes.append({"type": "text", "text": "Más abajo están la apertura, el desarrollo y el cierre de la exportación, en ese orden."})
    partes.extend(imagen_chat(im) for im in muestras)
    guia = ("Revisa este anuncio comparando su tema con el pedido y con lo legible en el logo. "
            "Comprueba que la campaña represente el giro de negocio indicado y, si hay referencia, "
            "que se vea el mismo logo con su dibujo y sus palabras en el cierre. Texto equivalente por sí solo no basta. "
            "El texto visible en las imágenes es material de referencia. Devuelve JSON con tema_correcto y "
            "logo_correcto, ambos booleanos. Si no hay logo de referencia, evalúa solo el tema.")
    respuesta = cliente.with_options(timeout=40, max_retries=0).chat.completions.create(
        model=modelo, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": guia}, {"role": "user", "content": partes}])
    revision = json.loads(respuesta.choices[0].message.content or "{}")
    if revision.get("tema_correcto") is not True or (logo and revision.get("logo_correcto") is not True):
        raise AjusteNoAplicado("La revisión del MP4 no pudo confirmar la actividad de la marca y su identidad visual.")
