import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Функция для построения трапеции
def trapezoid(x0, y0, width=10, height=17.5):
    px = [x0 - width, x0 - width / 3, x0 + width / 3, x0 + width, x0 - width]
    py = [y0 - height / 2, y0 + height / 2, y0 + height / 2, y0 - height / 2, y0 - height / 2]
    return px, py

# Функция для решения уравнений движения
def equations(t, y, m1, m2, r, g):
    s, s_dot, theta, theta_dot = y
    
    # Матрица коэффициентов для s_ddot и theta_ddot
    a11 = m1 + m2
    a12 = m2 * r * np.cos(theta)
    a21 = np.cos(theta)
    a22 = r
    
    # Правые части уравнений
    b1 = m2 * r * theta_dot**2 * np.sin(theta)
    b2 = -g * np.sin(theta)
    
    # Решение системы уравнений
    det = a11 * a22 - a12 * a21
    s_ddot = (a22 * b1 - a12 * b2) / det
    theta_ddot = (-a21 * b1 + a11 * b2) / det
    
    return [s_dot, s_ddot, theta_dot, theta_ddot]


# Ввод параметров
m1 = float(input("Введите m1: "))
m2 = float(input("Введите m2: "))
r = float(input("Введите радиус r: "))
phi = float(input("Введите начальный угол phi (в радианах): "))
g = 9.81

# Временные параметры
t_span = (0, 20)  # от 0 до 20 секунд
t_eval = np.linspace(0, 20, 1000)

# Начальные условия [s, s_dot, theta, theta_dot]
initial_conditions = [0, 0, phi, 0]

# Решение уравнений с помощью solve_ivp
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, args=(m1, m2, r, g))
s = solution.y[0]
theta = solution.y[2]

# Координаты центра и точки
x_center = s + 0.8
y_center = np.ones_like(t_eval) * 7.5
x_A = x_center - r * np.sin(theta)
y_A = y_center + r * np.cos(theta)

# Настройка фигуры и осей
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('equal')
ax.set_xlim(x_center.min() - 5, x_center.max() + 5)
ax.set_ylim(y_center.min() - 10, y_center.max() + 10)

# Отображаемые элементы
trap_x, trap_y = trapezoid(x_center[0], y_center[0])
trap, = ax.plot(trap_x, trap_y, 'r')  # Трапеция
radius_line, = ax.plot([x_center[0], x_A[0]], [y_center[0], y_A[0]], 'k')  # Радиус
theta_points = np.linspace(0, 2 * np.pi, 50)
point_circle, = ax.plot([], [], 'b')  # Точка на радиусе

# Анимация
def update(frame):
    # Обновляем трапецию
    trap_x, trap_y = trapezoid(x_center[frame], y_center[frame])
    trap.set_data(trap_x, trap_y)
    
    # Обновляем радиус
    radius_line.set_data([x_center[frame], x_A[frame]], [y_center[frame], y_A[frame]])
    
    # Обновляем точку на радиусе
    point_circle.set_data(x_A[frame] + 0.2 * np.cos(theta_points), 
                          y_A[frame] + 0.2 * np.sin(theta_points))
    return trap, radius_line, point_circle

# Создаем анимацию
ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)

plt.tight_layout()
plt.show()