def limpiar_datos_diagnosis(datos):
    # Normalizamos los datos
    data = [[1 if item == 'M' else 0 if item == 'B' else item for item in sublist] for sublist in datos]
    
    # Quitamos la columna ID
    data = quitar_columna(data, 0)
    return data

def quitar_columna(datos, columna):
    for fila in datos:
        fila.pop(columna)
    return datos