import numpy as np
import matriz as mt


def generar_biclases(datos, columan_clase, guardar_clase=False):
    A = []
    B = []
    for dato in datos:
        clase = dato[columan_clase] if guardar_clase else dato.pop(columan_clase)
        if clase == 1:
            A.append(dato)
        else:
            B.append(dato)
    return A, B


def promedio_clase(datos):
    promedios = []
    for i in range(len(datos[0])):
        suma = sum(fila[i] for fila in datos)
        promedios.append(suma / len(datos))
    return promedios


def matriz_covarianza_muestrales(datos, promedios):
    matriz_covarianza = mt.llenar_matriz(len(promedios), len(promedios))
    n = len(datos[0])
    for y in range(0, n):
        for x in range(y, n):
            if y == x:
                s_nn = sum((fila[y]-promedios[y])
                           ** 2 for fila in datos)/len(datos)
                matriz_covarianza[y][x] = s_nn
            else:
                s_nm = sum((fila[y]-promedios[y])*(fila[x]-promedios[x])
                           for fila in datos)/len(datos)
                matriz_covarianza[y][x] = s_nm
                matriz_covarianza[x][y] = s_nm

    return matriz_covarianza


def Sp(covarianza_A, num_elem_A, covarianza_B, num_elem_B):
    resultado = mt.suma(
        mt.punto(covarianza_A, num_elem_A),
        mt.punto(covarianza_B, num_elem_B)
    )
    resultado = mt.punto(
        resultado, (1/(num_elem_A+num_elem_B-2)))
    return resultado


def funcion_lineal(propuesta, promedios_a, promedios_b, inverso):
    # Sumatorias de promedios
    promedios = [promedios_a, promedios_b]
    sumatorias_promedios = []
    restas_promedios = []
    
    for i in range(len(promedios[0])):
        columna = [fila[i] for fila in promedios]
        sumatorias_promedios.append(sum(columna))
        restas_promedios.append(columna[0] - sum(columna[1:]))

    propuesta = np.array(propuesta)
    sumatorias_promedios = np.array(sumatorias_promedios)
    restas_promedios = np.array(restas_promedios)
    
    L_x = (propuesta-((1/2)*sumatorias_promedios)).dot(inverso).dot(restas_promedios.T)

    return L_x

def fisher(propuesta, grupo_A, grupo_B, to_print=False):
    # Cálculo de promedios
    promedios_grupo_A = promedio_clase(grupo_A)
    promedios_grupo_B = promedio_clase(grupo_B)

    covarianza_grupo_A = matriz_covarianza_muestrales(grupo_A, promedios_grupo_A)
    covarianza_grupo_B = matriz_covarianza_muestrales(grupo_B, promedios_grupo_B)

    s_p = Sp(
        covarianza_A=covarianza_grupo_A,
        covarianza_B=covarianza_grupo_B,
        num_elem_A=len(grupo_A),
        num_elem_B=len(grupo_B)
    )

    inversa = mt.inversa(s_p)
    
    L_x = funcion_lineal(propuesta, promedios_grupo_A, promedios_grupo_B, inversa)
    
    if to_print:
        print("Promedios Grupo A")
        mt.mostrar_matriz(promedios_grupo_A)
        print("Promedios Grupo B")
        mt.mostrar_matriz(promedios_grupo_B)
        print("Covarianza Grupo A")
        mt.mostrar_matriz(covarianza_grupo_A)
        print("Covarianza Grupo B")
        mt.mostrar_matriz(covarianza_grupo_B)
        print("Sp")
        mt.mostrar_matriz(s_p)
        print("Inversa de Sp")
        mt.mostrar_matriz(inversa)
    
    return L_x
