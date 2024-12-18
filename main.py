import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Функция для построения трапеции
def trapezoid(x0, y0, width=10, height=17.5):
    px = [x0 - width, x0 - width / 3, x0 + width / 3, x0 + width, x0 - width]
    py = [y0 - height / 2, y0 + height / 2, y0 + height / 2, y0 - height / 2, y0 - height / 2]
    return px, py

# Ввод параметров с клавиатуры
m1 = float(input("Введите массу m1: "))
m2 = float(input("Введите массу m2: "))
r = float(input("Введите длину радиуса r: "))
phi_deg = float(input("Введите начальный угол phi (в градусах): "))
phi = np.radians(phi_deg)  # переводим угол в радианы

# Константа
g = 9.81   # ускорение свободного падения

# Уравнения движения
def equations(t, y):
    s, s_dot, theta, theta_dot = y
    # Решение системы уравнений
    s_ddot = (-m2 * r * (theta_dot**2 * np.sin(theta))) / (m1 + m2)
    theta_ddot = -(s_ddot * np.cos(theta) + g * np.sin(theta)) / r
    return [s_dot, s_ddot, theta_dot, theta_ddot]

# Начальные условия: s, s_dot, theta, theta_dot
y0 = [0, 0, phi, 0]

# Временные параметры
t_span = (0, 20)  # от 0 до 20 секунд
t_eval = np.linspace(*t_span, 1000)

# Решаем систему уравнений
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)
s, _, theta, _ = sol.y

# Координаты центра
x_center = s + 0.8
y_center = np.ones_like(t_eval) * 7.5

# Радиус точки
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
theta_circle = np.linspace(0, 2 * np.pi, 50)
point_circle, = ax.plot([], [], 'b')  # Точка на радиусе

# Анимация
def update(frame):
    # Обновляем трапецию
    trap_x, trap_y = trapezoid(x_center[frame], y_center[frame])
    trap.set_data(trap_x, trap_y)
    
    # Обновляем радиус
    radius_line.set_data([x_center[frame], x_A[frame]], [y_center[frame], y_A[frame]])
    
    # Обновляем точку на радиусе
    point_circle.set_data(x_A[frame] + 0.2 * np.cos(theta_circle), y_A[frame] + 0.2 * np.sin(theta_circle))
    return trap, radius_line, point_circle

# Создаем анимацию
ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)

plt.tight_layout()
plt.show()
