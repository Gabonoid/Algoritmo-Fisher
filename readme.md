# Programa 6
## Primera Parte
Para este programa se utilizará la base de datos de cáncer de seno wdbc.data (Breast Cancer) que pueden descargar desde la web de UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
1. Lo que tiene que hacer este programa es calcular la función L(x) para nuevos datos, aplicando el criterio de mayor o menor que cero para clasificar un nuevo registro. Lo que tiene que introducir el usuario es un nuevo registro y que el programa diga si ese nuevo registro lo clasifica como un caso de tumor maligno o tumor benigno.
2. Poner a prueba su programa con los datos de los vinos que vimos en clase para ver si están haciendo bien sus cálculos.

## Segunda Parte
1. Así mismo para este programa y estos datos, tienen que calcular la matriz de confusión eligiendo un conjuntos de datos de prueba y otro de entrenamiento (puede ser un 10% y un 90% de los datos respectivamente).
2. Utilizan los datos de entrenamiento para calcular la L(x) y después, en esta función L(x), comienzan a introducir los datos de prueba para ver, en cada uno de ellos, si clasificó bien o no y de esta forma comenzar a contar el número de verdaderos (positivos y negativos) y de falsos (positivos y negativos).
3. Teniendo la matriz de confusión calculan la exactitud, la precisión y la sensibilidad para decir si discriminante lineal de fisher es o no una buena metodología de clasificación para esta base de datos del cáncer.