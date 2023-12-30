import pandas as pd
from helpers import limpieza
import fisher as fh
import clasificacion as cl

import matriz as mt
# Primera Parte
print('PRIMERA PARTE'.center(50, '-'))

# Parte 1-1
print('Parte 1-1'.center(50, '-'))

# Lectura de datos
cancer_datos = pd.read_csv('data/wdbc.csv', header=None).values.tolist()

# Limpieza de datos
cancer_datos = limpieza.limpiar_datos_diagnosis(cancer_datos)

# Generación de clases
malignos, benignos = fh.generar_biclases(cancer_datos, 0)

# Generación de propuesta
# Benigno
propuesta = [9.173, 13.86, 59.2, 260.9, 0.07721, 0.08751, 0.05988, 0.0218, 0.2341, 0.06963, 0.4098, 2.265, 2.608, 23.52, 0.008738,0.03938, 0.04312, 0.0156, 0.04192, 0.005822, 10.01, 19.23, 65.59, 310.1, 0.09836, 0.1678, 0.1397, 0.05087, 0.3282, 0.0849]

''' # Maligno
propuesta = [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189] '''

L_x_cancer = fh.fisher(propuesta, malignos, benignos)
print("L(x) =", L_x_cancer)
print(f"Los valores propuestos pertenecen a la clase {'Maligno' if L_x_cancer > 0 else 'Benigno'}")


# Parte 1-2
print('Parte 1-2'.center(50, '-'))
# Poniendo a prueba el cloritmo con los datos visto en clase
vino_1 = pd.read_csv('data/vino_1.csv', header=None).values.tolist()
vino_2 = pd.read_csv('data/vino_2.csv', header=None).values.tolist()

propuesta_vino = [13.05, 515]
L_x_vino = fh.fisher(propuesta_vino, vino_1, vino_2)
print("L(x) =", L_x_vino)
print(
    f"Los valores {propuesta_vino} pertenecen a la clase Vino {1 if L_x_vino > 0 else 2}")


# Segunda Parte
print('SEGUNDA PARTE'.center(50, '-'))

# Generamos las Clases
cancer_datos = pd.read_csv('data/wdbc.csv', header=None).values.tolist()
cancer_datos = limpieza.limpiar_datos_diagnosis(cancer_datos)
entrenamiento, prueba = cl.generar_pruebas(0.1, cancer_datos, guardar=False, is_random=False)

conjunto = cl.generar_conjunto(entrenamiento, prueba)
print(conjunto)
print("Matriz de confusion".center(50, "-"))
vp, vn, fp, fn = cl.clases(conjunto)

print("Verdaderos Positivos (VP): ", vp)
print("Verdaderos Negativos (VN): ", vn)
print("Falsos Positivos (FP): ", fp)
print("Falsos Negativos (FN): ", fn)

exactitud = cl.exactitud(vp, vn, fp, fn)
print("Exactitud:", exactitud, "")

precision = cl.precision(vp, fp)
print("Precision:", precision)

sensibilidad = cl.sensibilidad(vp, fn)
print("Sensibilidad:", sensibilidad)

tnr = cl.razon_verdaderos_negativos(vn, fp)
print("TNR:", tnr)

fpr = cl.razon_falsos_positivos(fp, vn)
print("FPR:", fpr)

print(
    f"La función de clasificación tuvo un {round(exactitud*100, 2)}% de exactitud por lo que es un {'buen' if exactitud > 0.85 else 'mal'} clasificador.")
