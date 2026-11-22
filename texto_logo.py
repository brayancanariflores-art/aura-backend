"""Sustitución de una leyenda corta en el arte antes de proyectarlo sobre el video."""
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

from ajustes_marca import AjusteNoAplicado
from seguimiento_marca import imagen_chat


def cambio_de_leyenda(orden):
    # Las comillas delimitan las dos cadenas sin confundir un cambio de marca por otro archivo.
    frase = r'[«“"]([^»”"\n]{1,64})[»”"]'
    patron = r'\b(?:reemplaz\w*|sustitu\w*|cambi\w*)\s+(?:(?:la|el)\s+(?:palabra|texto|leyenda)\s+)?'
    par = re.search(patron + frase + r'\s+por\s+' + frase, orden, re.I)
    if par:
        return par.groups()
    simple = re.search(r"\b(?:reemplaz\w*|sustitu\w*|cambi\w*)\s+(?:la palabra|el texto|la leyenda)\s+"
                       r"([\w -]{1,45}?)\s+por\s+([\w -]{1,45}?)(?=\s+(?:en el|encima|debajo|y (?:mant|conserv|centr))|[.,;!]|$)",orden,re.I)
    return tuple(c.strip() for c in simple.groups()) if simple else None


def leer_leyenda(im, origen, destino, cliente, modelo):
    foto = im.copy()
    foto.thumbnail((850,850),Image.Resampling.LANCZOS)
    tarea = ('Lee la leyenda de la imagen. Devuelve JSON con estado "origen", "destino" o "ausente". '
             'Si el texto origen está visible indica su caja apretada incluyendo acentos, en milésimas '
             'de la imagen completa, como "caja":[izquierda,arriba,derecha,abajo]. '
             'Si solo ves el destino indica "destino". Ignora mayúsculas y minúsculas. '
             'No incluyas otras líneas en la caja. Indica con "mayusculas":true si toda la leyenda va en mayúsculas.')
    # Componer transparencias para que un fondo negro implícito no oculte la tinta oscura.
    vista = Image.new('RGB',foto.size,'white')
    vista.paste(foto,mask=foto.getchannel('A'))
    resp=cliente.with_options(timeout=35,max_retries=1).chat.completions.create(model=modelo,
        response_format={'type':'json_object'},messages=[{'role':'system','content':tarea},
        {'role':'user','content':[{'type':'text','text':json.dumps({'origen':origen,'destino':destino},ensure_ascii=False)},imagen_chat(vista)]}])
    return json.loads(resp.choices[0].message.content or '{}')


def pintar_leyenda(im, caja, texto):
    dims=np.array([im.width,im.height,im.width,im.height])
    if not isinstance(caja,list) or len(caja)!=4 or any(type(k) not in (int,float) or not 0<=k<=1000 for k in caja):
        raise AjusteNoAplicado('No pude delimitar la palabra en el logo.')
    x,y,r,b=np.rint(dims*np.array(caja)/1000).astype(int)
    if r-x<5 or b-y<4 or y<2 or b+2>=im.height or (r-x)*(b-y)>.4*im.width*im.height:
        raise AjusteNoAplicado('No pude aislar la leyenda del resto del logo.')
    # Ajustar la caja a los trazos reales sobre un fondo liso de color o transparente.
    pix=np.array(im)
    zona=pix[y:b,x:r].astype(float)
    arriba=np.median(pix[y-2:y,x:r],axis=0)
    abajo=np.median(pix[b:b+2,x:r],axis=0)
    f=np.linspace(0,1,b-y)[:,None,None]
    fondo=arriba*(1-f)+abajo*f
    def visible(a):
        return a[:,:,:3]*(a[:,:,3:4]/255)+255-a[:,:,3:4]
    contraste=np.linalg.norm(visible(zona)-visible(fondo),axis=2)
    umbral=max(22.,float(np.percentile(contraste,90))*.48)
    trazos=contraste>umbral
    if trazos.sum()<4 or trazos.mean()>.82:
        raise AjusteNoAplicado('El contraste de la leyenda es insuficiente para cambiarla con precisión.')
    sy,sx=np.where(trazos)
    tinta=np.median(zona[contraste>=np.percentile(contraste[trazos],65)],axis=0)
    borrar=ndimage.maximum_filter(trazos,size=3)
    pix[y:b,x:r][borrar]=np.rint(fondo[borrar]).clip(0,255).astype('uint8')
    ancho=max(1,sx.max()-sx.min()+1); alto=max(1,sy.max()-sy.min()+1)
    font=ImageFont.load_default(size=112)
    try:
        font=ImageFont.truetype('DejaVuSans-Bold.ttf',size=112)
    except OSError:
        pass
    l,t,rr,bb=font.getbbox(texto)
    letras=Image.new('L',(max(1,rr-l),max(1,bb-t)))
    ImageDraw.Draw(letras).text((-l,-t),texto,fill=255,font=font)
    factor=min(ancho/letras.width,alto/letras.height)
    letras=letras.resize((max(1,round(letras.width*factor)),max(1,round(letras.height*factor))),Image.Resampling.LANCZOS)
    resultado=Image.fromarray(pix)
    pos=(x+sx.min()+(ancho-letras.width)//2,y+sy.min()+(alto-letras.height)//2)
    resultado.paste(tuple(map(int,tinta)),(*pos,pos[0]+letras.width,pos[1]+letras.height),letras)
    return resultado


def preparar_leyenda(carpeta, orden, cliente, modelo):
    derivado=carpeta/'logo-editado.png'
    ruta=derivado if derivado.is_file() else carpeta/'marca.png'
    par=cambio_de_leyenda(orden)
    if not par or par[0].casefold()==par[1].casefold():
        return ruta
    with Image.open(ruta) as fuente:
        im=ImageOps.exif_transpose(fuente).convert('RGBA')
    old,new=par
    medida=leer_leyenda(im,old,new,cliente,modelo)
    if medida.get('estado')=='destino':
        return ruta
    if medida.get('estado')!='origen':
        raise AjusteNoAplicado('No encontré la palabra indicada en el logo adjunto.')
    corregido=pintar_leyenda(im,medida.get('caja'),new.upper() if medida.get('mayusculas') is True else new)
    if leer_leyenda(corregido,old,new,cliente,modelo).get('estado')!='destino':
        raise AjusteNoAplicado('No pude verificar la nueva leyenda en el logo. Revisa la imagen de la marca.')
    corregido.save(derivado)
    return derivado
