# Descenso por gradiente en una red de tres neuronas

**Fundamentos matemáticos y estadísticos para IA**

El proyecto muestra cómo aplicar la regla de la cadena para calcular el gradiente de una red neuronal y cómo usarlo para ajustar sus pesos y sesgos. La red aprende la función XOR a partir de sus cuatro combinaciones de entrada.

## Ejecutar el proyecto

Descomprime el ZIP, abre una terminal en la carpeta `Proyecto_Red_3_Neuronas` y ejecuta:

```bash
python3 Gradiente_Descendente_Red_3_Neuronas.py
```

Solo se requiere Python 3. En Windows puedes usar `py -3` en lugar de `python3`.

| Archivo | Contenido |
| --- | --- |
| `Gradiente_Descendente_Red_3_Neuronas.py` | Datos XOR, propagación, retropropagación, verificación del gradiente y entrenamiento |
| `README.md` | Instrucciones de ejecución y explicación matemática |

La ejecución predeterminada usa una tasa de aprendizaje de `0.1` y realiza `2000` actualizaciones. Puedes cambiar estos valores desde la terminal.

```bash
python3 Gradiente_Descendente_Red_3_Neuronas.py --tasa 0.1 --epocas 1
```

## Arquitectura 2–2–1

Cada ejemplo tiene dos entradas, `x1` y `x2`. La red tiene dos neuronas ocultas con activación tangente hiperbólica y una neurona de salida lineal. En total se entrenan **seis pesos y tres sesgos**.

| Capa | Cálculo | Parámetros |
| --- | --- | --- |
| Neurona oculta 1 | `z1 = w11*x1 + w12*x2 + b1`, `h1 = tanh(z1)` | `w11`, `w12`, `b1` |
| Neurona oculta 2 | `z2 = w21*x1 + w22*x2 + b2`, `h2 = tanh(z2)` | `w21`, `w22`, `b2` |
| Neurona de salida | `prediccion = v1*h1 + v2*h2 + b3` | `v1`, `v2`, `b3` |

XOR tiene objetivo `1` cuando las entradas son diferentes y `0` cuando son iguales. La salida lineal produce un valor real que el entrenamiento acerca a esos objetivos. Durante el ajuste pueden aparecer predicciones fuera del intervalo `[0, 1]`.

## Cómo aprende la red

1. **Propagación hacia adelante.** `propagacion()` calcula las activaciones ocultas y la predicción para una entrada.
2. **Pérdida.** Para cada observación se calcula el residuo `e = prediccion - objetivo` y la pérdida individual `e**2 / 2`. `perdida()` promedia las pérdidas de las cuatro observaciones.
3. **Retropropagación.** `retropropagacion()` calcula las nueve derivadas parciales de la pérdida individual. La salida lineal aporta `delta3 = e`. Como la derivada de `tanh(z)` es `1 - tanh(z)**2`, las señales de error ocultas se obtienen mediante la regla de la cadena.
4. **Gradiente del lote.** `gradiente_lote()` promedia las derivadas parciales de las cuatro observaciones para obtener el gradiente de la pérdida total.
5. **Actualización.** `actualizar()` aplica `parametro_nuevo = parametro_actual - tasa * gradiente` a los nueve parámetros. Todas las derivadas del paso se calculan a partir del mismo conjunto de parámetros. Cada época procesa las cuatro observaciones y realiza una actualización.

### De la regla de la cadena al código

Para calcular cuánto cambia la pérdida individual al variar `w11`, seguimos su efecto sobre `z1`, `h1`, la predicción y la pérdida.

```text
d(pérdida individual)/d(w11) = e * v1 * (1 - h1**2) * x1
                              = delta1 * x1
```

El peso `w11` recibe una contribución de cero de cualquier observación donde `x1 = 0`. Para `b1`, el factor final es `1`, pues `z1` cambia una unidad por cada unidad de cambio en `b1`. Por eso su derivada es `delta1`.

Antes del entrenamiento, el programa compara las nueve derivadas analíticas con una aproximación por diferencias finitas centradas. Para cada parámetro, aumenta y disminuye su valor en `epsilon = 1e-5` con los demás parámetros fijos, y divide la diferencia de pérdidas por `2 * epsilon`. El error máximo de esta comparación es aproximadamente `7.26e-12` con la configuración inicial.

## Interpretar la salida

La primera tabla muestra el valor inicial, el gradiente y el nuevo valor de cada parámetro. Por ejemplo, `w11` empieza en `0.500000` y su gradiente promedio es aproximadamente `-0.075489`. Al restar `0.1 * (-0.075489)`, el peso aumenta a `0.507549`.

Con los valores predeterminados, la pérdida pasa de **0.25378417** a **0.21678599** tras la primera actualización y llega aproximadamente a **0.00000007** en la época 2000.

| Entrada | Objetivo XOR | Predicción final aproximada |
| --- | ---: | ---: |
| (0, 0) | 0 | 0.000259 |
| (0, 1) | 1 | 0.999681 |
| (1, 0) | 1 | 0.999683 |
| (1, 1) | 0 | 0.000572 |

## Experimento para clase

Ejecuta un paso con `--tasa 0.1 --epocas 1` y luego con `--tasa 2 --epocas 1`. Antes de la segunda ejecución, predice si la pérdida aumentará o disminuirá. Compara las dos tablas y explica cómo es posible obtener pérdidas distintas partiendo del mismo gradiente.
