import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Функция для построения трапеции
def trapezoid(x0, y0, width=10, height=17.5):
    px = [x0 - width, x0 - width / 3, x0 + width / 3, x0 + width, x0 - width]
    py = [y0 - height / 2, y0 + height / 2, y0 + height / 2, y0 - height / 2, y0 - height / 2]
    return px, py

# Временные параметры
T = np.linspace(0, 20, 1000)

# Движение центра и угла
s = 4 * np.cos(3 * T)
phi = 4 * np.sin(T - 10)

# Координаты центра
x_center = -s
y_center = np.ones_like(T) * 7.5

# Радиус точки
radius_length = 3
x_A = x_center - radius_length * np.sin(phi)
y_A = y_center + radius_length * np.cos(phi)

# Настройка фигуры и осей
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('equal')
ax.set_xlim(x_center.min() - 10, x_center.max() + 10)
ax.set_ylim(y_center.min() - 15, y_center.max() + 15)

# Отображаемые элементы
trap_x, trap_y = trapezoid(x_center[0], y_center[0])
trap, = ax.plot(trap_x, trap_y, 'r')  # Трапеция
radius_line, = ax.plot([x_center[0], x_A[0]], [y_center[0], y_A[0]], 'k')  # Радиус
theta = np.linspace(0, 2 * np.pi, 100)
point_circle, = ax.plot([], [], 'bo')  # Точка на радиусе
orbit_circle, = ax.plot([], [], color="gray", linestyle="--")  # Окружность

# Анимация
def update(frame):
    # Обновляем трапецию
    trap_x, trap_y = trapezoid(x_center[frame], y_center[frame])
    trap.set_data(trap_x, trap_y)
    
    # Обновляем радиус
    radius_line.set_data([x_center[frame], x_A[frame]], [y_center[frame], y_A[frame]])
    
    # Обновляем точку на радиусе
    point_circle.set_data([x_A[frame]], [y_A[frame]])
    
    # Обновляем окружность
    orbit_x = x_center[frame] + radius_length * np.cos(theta)
    orbit_y = y_center[frame] + radius_length * np.sin(theta)
    orbit_circle.set_data(orbit_x, orbit_y)
    
    return trap, radius_line, point_circle, orbit_circle

# Создаем анимацию
ani = FuncAnimation(fig, update, frames=len(T), interval=20, blit=True)

plt.tight_layout()
plt.show()
