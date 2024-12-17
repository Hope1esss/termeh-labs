import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Define time points and symbolic time variable
time_points = np.linspace(0, 10, 1000)
radius = 1
t = sp.symbols('t')

# Parametric equations
r_expr = 1 - sp.sin(t)
fi_expr = 2 * t

# Convert to Cartesian coordinates
x_expr = radius * r_expr * sp.cos(fi_expr)
y_expr = radius * r_expr * sp.sin(fi_expr)

# Compute velocity (first derivative)
vx_expr = sp.diff(x_expr, t)
vy_expr = sp.diff(y_expr, t)

# Compute acceleration (second derivative)
ax_expr = sp.diff(vx_expr, t)
ay_expr = sp.diff(vy_expr, t)

# Convert to numpy functions for numerical evaluation
x_func = sp.lambdify(t, x_expr, 'numpy')
y_func = sp.lambdify(t, y_expr, 'numpy')
vx_func = sp.lambdify(t, vx_expr, 'numpy')
vy_func = sp.lambdify(t, vy_expr, 'numpy')
ax_func = sp.lambdify(t, ax_expr, 'numpy')
ay_func = sp.lambdify(t, ay_expr, 'numpy')

# Evaluate functions
x_vals = x_func(time_points)
y_vals = y_func(time_points)
vx_vals = vx_func(time_points)
vy_vals = vy_func(time_points)
ax_vals = ax_func(time_points)
ay_vals = ay_func(time_points)

# Set up plot
fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim([np.min(x_vals) - 0.8, np.max(x_vals) + 0.8])
ax.set_ylim([np.min(y_vals) - 0.8, np.max(y_vals) + 0.8])

# Initialize plot elements
point, = ax.plot([], [], 'go', markersize=10)
ax.plot(x_vals, y_vals, 'r-', lw=1)
velocity_line, = ax.plot([], [], 'b-', lw=1)
velocity_arrow_head, = ax.plot([], [], 'b-', lw=1)
acceleration_line, = ax.plot([], [], 'g-', lw=1)
acceleration_arrow_head, = ax.plot([], [], 'g-', lw=1)
radius_vector_line, = ax.plot([], [], 'y-', lw=1)
radius_vector_arrow_head, = ax.plot([], [], 'y-', lw=1)
curvature_radius_line, = ax.plot([], [], 'm--', lw=1)
curvature_radius_arrow_head, = ax.plot([], [], 'm--', lw=1)

def rotate_vectors(x_arr, y_arr, angle):
    x_new = x_arr * np.cos(angle) - y_arr * np.sin(angle)
    y_new = x_arr * np.sin(angle) + y_arr * np.cos(angle)
    return x_new, y_new

def update(frame):
    # Get the current position, velocity, acceleration, and other vectors
    x0 = x_vals[frame]
    y0 = y_vals[frame]
    vx = vx_vals[frame]
    vy = vy_vals[frame]
    ax0 = ax_vals[frame]
    ay0 = ay_vals[frame]

    # Update the position of the moving point
    point.set_data([x0], [y0])

    # Update velocity vector
    velocity_line.set_data([x0, x0 + vx], [y0, y0 + vy])
    velocity_angle = math.atan2(vy, vx)
    arrow_x = np.array([-0.08, 0, -0.08])
    arrow_y = np.array([0.04, 0, -0.04])
    VArrowX, VArrowY = rotate_vectors(arrow_x, arrow_y, velocity_angle)
    velocity_arrow_head.set_data(VArrowX + x0 + vx, VArrowY + y0 + vy)

    # Update acceleration vector
    acceleration_line.set_data([x0, x0 + ax0], [y0, y0 + ay0])
    acceleration_angle = math.atan2(ay0, ax0)
    AArrowX, AArrowY = rotate_vectors(arrow_x, arrow_y, acceleration_angle)
    acceleration_arrow_head.set_data(AArrowX + x0 + ax0, AArrowY + y0 + ay0)

    # Update radius vector
    radius_vector_line.set_data([0, x0], [0, y0])
    radius_angle = math.atan2(y0, x0)
    RArrowX, RArrowY = rotate_vectors(arrow_x, arrow_y, radius_angle)
    radius_vector_arrow_head.set_data(RArrowX + x0, RArrowY + y0)

    # Calculate curvature radius
    numerator = (vx**2 + vy**2)**1.5
    denominator = abs(vx * ay0 - vy * ax0)
    if denominator != 0:
        R_curv = numerator / denominator
    else:
        R_curv = np.inf

    # Calculate the center of the curvature
    norm_vx = -vy
    norm_vy = vx
    norm = np.hypot(norm_vx, norm_vy)
    if norm != 0:
        norm_vx /= norm
        norm_vy /= norm

    center_x = x0 + R_curv * norm_vx
    center_y = y0 + R_curv * norm_vy

    # Update curvature radius vector
    curvature_radius_line.set_data([x0, center_x], [y0, center_y])
    curvature_angle = math.atan2(center_y - y0, center_x - x0)
    CArrowX, CArrowY = rotate_vectors(arrow_x, arrow_y, curvature_angle)
    curvature_radius_arrow_head.set_data(CArrowX + center_x, CArrowY + center_y)

    return (point,
            velocity_line, velocity_arrow_head,
            acceleration_line, acceleration_arrow_head,
            radius_vector_line, radius_vector_arrow_head,
            curvature_radius_line, curvature_radius_arrow_head)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_points), interval=20, blit=True)

# Show the plot with animation
plt.show()
