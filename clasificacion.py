import random
import pandas as pd
import fisher as fh


def generar_pruebas(porcentaje, datos, guardar=False, is_random=False):
    porcentaje_promedio = int(len(datos) * porcentaje)
    
    if is_random:
        prueba = random.sample(datos, porcentaje_promedio)
        entrenamiento = [dato for dato in datos if dato not in prueba]
    else:
        prueba = datos[:porcentaje_promedio]
        entrenamiento = datos[porcentaje_promedio:]

    if guardar:
        pd.DataFrame(prueba).to_csv(
            "data/prueba.csv", index=False, header=False)
        pd.DataFrame(entrenamiento).to_csv(
            "data/entrenamiento.csv", index=False, header=False)

    return entrenamiento, prueba


def generar_conjunto(datos_entrenamiento, datos_prueba):
    finales_prueba = [dato.pop(0) for dato in datos_prueba]
    finales_entrenamiento = []

    malignos, benignos = fh.generar_biclases(datos_entrenamiento, 0)

    for dato in datos_prueba:
        L_x_cancer = fh.fisher(dato, malignos, benignos)
        finales_entrenamiento.append(1 if L_x_cancer > 0 else 0)

    return [[i, j]for i, j in zip(finales_prueba, finales_entrenamiento)]


def clases(matriz_confusion):
    vp = 0
    vn = 0
    fp = 0
    fn = 0
    for linea in matriz_confusion:
        if linea[0] == 1 and linea[1] == 1:
            vp += 1
        elif linea[0] == 1 and linea[1] == 0:
            fn += 1
        elif linea[0] == 0 and linea[1] == 1:
            fp += 1
        elif linea[0] == 0 and linea[1] == 0:
            vn += 1
    return vp, vn, fp, fn


def exactitud(vp, vn, fp, fn):
    return (vp + vn) / (vp + vn + fp + fn)


def sensibilidad(vp, fn):
    return vp / (vp + fn)


def precision(vp, fp):
    return vp / (vp + fp)


def razon_verdaderos_negativos(vn, fp):
    return vn / (vn + fp)


def razon_falsos_positivos(fp, vn):
    return fp / (fp + vn)


def razon_falsos_negativos(fn, vp):
    return fn / (fn + vp)
