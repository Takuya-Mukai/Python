import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib

N = 100
dt = 0.01
dx = 0.01

def u(x, t):
    if x >= 1:
        return u(x-1, t)
    else:
        return (np.sin(np.pi*x))**100

def central_difference(x, dt, dx):
    x_new = x.copy()
    for i in range(1, len(x) - 1):
        x_new[i] = x[i] - (0.5 * dt / dx) * (x[i + 1] - x[i - 1])
    return x_new

def upwind_difference(x, dt, dx):
    x_new = x.copy()
    for i in range(1, len(x) - 1):
        x_new[i] = x[i] - (dt / dx) * (x[i] - x[i - 1])
    return x_new

def runge_kutta(x, dt, dx, scheme):
    k1 = scheme(x, dt, dx)
    k2 = scheme(x + 0.5 * dt * k1, dt, dx)
    k3 = scheme(x + 0.5 * dt * k2, dt, dx)
    k4 = scheme(x + dt * k3, dt, dx)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

x_values = np.zeros(N+1)

for i in range(N+1):
    x_values[i] = u(i/N, 0)

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2)  # 視覚的な安定性のためにy軸の範囲を適切に設定

line, = ax.plot(np.linspace(0, 1, N+1), x_values)

def update(frame):
    global x_values
    x_values = runge_kutta(x_values, dt, dx, upwind_difference)
    line.set_ydata(x_values)
    return line,

ani = FuncAnimation(fig, update, frames=1000, interval=100, blit=True)

path_dir = pathlib.Path('test')
path_dir.mkdir(parents=True, exist_ok=True)
path_gif = path_dir.joinpath('animation-hist.gif')

ani.save(path_gif, writer='pillow')
plt.show()
