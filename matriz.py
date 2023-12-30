import numpy as np


def llenar_matriz(fila, columna, valor=0):
    return [[valor] * columna for _ in range(fila)]


def punto(matriz, valor):
    return [[elemento * valor for elemento in fila] for fila in matriz]


def suma(matriz1, matriz2):
    return [[matriz1[i][j] + matriz2[i][j] for j in range(len(matriz1[0]))] for i in range(len(matriz1))]


def resta(matriz1, matriz2):
    return [[matriz1[i][j] - matriz2[i][j] for j in range(len(matriz1[0]))] for i in range(len(matriz1))]


def inversa(matriz):
    inversa = np.linalg.inv(matriz)
    return inversa.tolist()


def transpuesta(matriz):
    return [[matriz[i][j] for i in range(len(matriz))] for j in range(len(matriz[0]))]


def mostrar_matriz(matriz):
    print(np.array(matriz), "\n")

