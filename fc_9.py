
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definimos el modelo de Lotka-Volterra
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Parámetros del modelo
alpha = 1.1   # Tasa de crecimiento de las presas
beta = 0.4    # Tasa a la cual los depredadores comen presas
delta = 0.1   # Tasa a la cual los depredadores se reproducen
gamma = 0.4   # Tasa de muerte de los depredadores

# Condiciones iniciales
x0 = 10   # Población inicial de presas
y0 = 5    # Población inicial de depredadores
z0 = [x0, y0]

# Tiempo de simulación
t_span = (0, 200)
t_eval = np.linspace(*t_span, 10000)

# Resolver el sistema de ecuaciones diferenciales
sol = solve_ivp(lotka_volterra, t_span, z0, args=(alpha, beta, delta, gamma), t_eval=t_eval)

# Extraer soluciones
t = sol.t
x = sol.y[0]
y = sol.y[1]

# Crear una sección de Poincaré
def poincare_section(x, y, y_threshold):
    points = []
    for i in range(1, len(y)):
        if y[i-1] < y_threshold and y[i] >= y_threshold:
            points.append(x[i])
    return points

# Umbral para la sección de Poincaré
y_threshold = 5

# Calcular la sección de Poincaré
poincare_points = poincare_section(x, y, y_threshold)

# Graficar la solución y la sección de Poincaré
plt.figure(figsize=(14, 6))

# Gráfica de las soluciones
plt.subplot(1, 2, 1)
plt.plot(t, x, label='Presas (x)')
plt.plot(t, y, label='Depredadores (y)')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Modelo de Lotka-Volterra')
plt.legend()
plt.grid(True)

# Gráfica de la sección de Poincaré
plt.subplot(1, 2, 2)
plt.scatter(poincare_points, [y_threshold]*len(poincare_points), color='red')
plt.xlabel('Población de presas (x)')
plt.ylabel('Umbral de población de depredadores (y)')
plt.title('Sección de Poincaré')
plt.grid(True)

plt.tight_layout()
plt.show()