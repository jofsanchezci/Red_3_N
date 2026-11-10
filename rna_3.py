"""Descenso por gradiente para una red 2-2-1 entrenada con XOR.

Dos entradas alimentan dos neuronas ocultas tanh y una neurona de salida
lineal. Las entradas son datos. La red tiene tres neuronas entrenables
y nueve parámetros. Se usa únicamente la biblioteca estándar de Python.
"""

import argparse
import math


DATOS = (
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
)


def propagacion(entradas, p):
    """Devuelve las dos activaciones ocultas y la predicción."""
    x1, x2 = entradas
    z1 = p["w11"] * x1 + p["w12"] * x2 + p["b1"]
    z2 = p["w21"] * x1 + p["w22"] * x2 + p["b2"]
    h1 = math.tanh(z1)
    h2 = math.tanh(z2)
    prediccion = p["v1"] * h1 + p["v2"] * h2 + p["b3"]
    return h1, h2, prediccion


def perdida(datos, p):
    """Calcula L = suma de residuos al cuadrado / (2*n)."""
    return sum(
        (propagacion(entradas, p)[2] - objetivo) ** 2
        for entradas, objetivo in datos
    ) / (2 * len(datos))


def retropropagacion(entradas, objetivo, p):
    """Gradientes de la pérdida de una observación respecto a los pesos y sesgos."""
    x1, x2 = entradas
    h1, h2, prediccion = propagacion(entradas, p)

    # Salida lineal y pérdida individual (prediccion - objetivo)^2 / 2.
    delta3 = prediccion - objetivo

    # Regla de la cadena en las ocultas. Si h = tanh(z), dh/dz = 1 - h^2.
    delta1 = delta3 * p["v1"] * (1 - h1**2)
    delta2 = delta3 * p["v2"] * (1 - h2**2)

    return {
        "w11": delta1 * x1,
        "w12": delta1 * x2,
        "b1": delta1,
        "w21": delta2 * x1,
        "w22": delta2 * x2,
        "b2": delta2,
        "v1": delta3 * h1,
        "v2": delta3 * h2,
        "b3": delta3,
    }


def gradiente_lote(datos, p):
    """Promedia las contribuciones de todas las observaciones."""
    acumulado = dict.fromkeys(p, 0.0)
    for entradas, objetivo in datos:
        for nombre, derivada in retropropagacion(entradas, objetivo, p).items():
            acumulado[nombre] += derivada
    return {nombre: suma / len(datos) for nombre, suma in acumulado.items()}


def error_del_gradiente(datos, p, gradiente, epsilon=1e-5):
    """Contrasta el gradiente analítico con diferencias finitas centradas."""
    errores = []
    for nombre, valor in p.items():
        adelante = {**p, nombre: valor + epsilon}
        atras = {**p, nombre: valor - epsilon}
        numerico = (perdida(datos, adelante) - perdida(datos, atras)) / (2 * epsilon)
        errores.append(abs(gradiente[nombre] - numerico))
    return max(errores)


def actualizar(p, gradiente, tasa):
    """Aplica un paso simultáneo con el gradiente del estado anterior."""
    return {nombre: valor - tasa * gradiente[nombre] for nombre, valor in p.items()}


def main():
    parser = argparse.ArgumentParser(description="Entrena una red de tres neuronas con XOR.")
    parser.add_argument("--tasa", type=float, default=0.1, help="Tasa de aprendizaje")
    parser.add_argument("--epocas", type=int, default=2000, help="Número de pasos con el lote completo")
    args = parser.parse_args()
    if not math.isfinite(args.tasa) or args.tasa <= 0:
        parser.error("--tasa debe ser un número finito y positivo")
    if args.epocas < 1:
        parser.error("--epocas debe ser al menos 1")

    # Inicio fijo y asimétrico para que ambas ocultas puedan aprender funciones distintas.
    p = {
        "w11": 0.5, "w12": 0.3, "b1": -0.2,
        "w21": -0.4, "w22": 0.6, "b2": 0.1,
        "v1": 0.7, "v2": -0.5, "b3": 0.0,
    }
    gradiente = gradiente_lote(DATOS, p)
    error = error_del_gradiente(DATOS, p, gradiente)
    siguiente = actualizar(p, gradiente, args.tasa)

    print("Red 2-2-1 | dos entradas, dos neuronas ocultas y una de salida")
    print(f"Error máximo entre gradiente analítico y numérico: {error:.2e}")
    print(f"\nPrimera actualización con tasa = {args.tasa:g}")
    print(f"{'Parámetro':<10} {'Inicial':>10} {'Gradiente':>10} {'Actualizado':>12}")
    for nombre, valor in p.items():
        print(f"{nombre:<10} {valor:>10.6f} {gradiente[nombre]:>10.6f} {siguiente[nombre]:>12.6f}")

    print("\nEvolución de la pérdida")
    print(f"Época {0:>4}: {perdida(DATOS, p):.8f}")
    p = siguiente
    intervalo = max(1, args.epocas // 5)
    for epoca in range(1, args.epocas + 1):
        try:
            valor = perdida(DATOS, p)
        except OverflowError:
            parser.exit(1, f"El entrenamiento diverge en la época {epoca}. Prueba una tasa menor.\n")
        if not math.isfinite(valor):
            parser.exit(1, f"El entrenamiento diverge en la época {epoca}. Prueba una tasa menor.\n")
        if epoca == 1 or epoca % intervalo == 0 or epoca == args.epocas:
            print(f"Época {epoca:>4}: {valor:.8f}")
        if epoca < args.epocas:
            p = actualizar(p, gradiente_lote(DATOS, p), args.tasa)

    print("\nPredicciones finales")
    for entradas, objetivo in DATOS:
        prediccion = propagacion(entradas, p)[2]
        print(f"Entrada {entradas} | objetivo {objetivo:g} | predicción {prediccion:.6f}")


if __name__ == "__main__":
    main()
