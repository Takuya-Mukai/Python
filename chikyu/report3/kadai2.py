import numpy as np
import matplotlib.pyplot as plt

# 初期条件
def initial_condition(x):
    return np.sin(np.pi * x) ** 100

# 境界条件
def boundary_condition(u):
    u[0] = u[-1]  # 周期境界条件

# 中心差分法
def central_difference(u, dt, dx):
    u_new = u.copy()
    for i in range(1, len(u) - 1):
        u_new[i] = u[i] - 0.5 * dt / dx * (u[i + 1] - u[i - 1])
    boundary_condition(u_new)
    return u_new

# 風上差分法
def upwind_difference(u, dt, dx):
    u_new = u.copy()
    for i in range(1, len(u) - 1):
        u_new[i] = u[i] - dt / dx * (u[i] - u[i - 1])
    boundary_condition(u_new)
    return u_new

# Runge-Kutta法
def runge_kutta(u, dt, dx, scheme):
    k1 = scheme(u, dt, dx)
    k2 = scheme(u + 0.5 * dt * k1, dt, dx)
    k3 = scheme(u + 0.5 * dt * k2, dt, dx)
    k4 = scheme(u + dt * k3, dt, dx)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# 計算パラメータ
L = 1.0  # 空間の長さ
T = 1.0  # 計算終了時刻
Nx = 100  # 空間方向の分割数
dt = 0.01  # 時間刻み幅 (CFL条件を考慮して調整)
dx_values = [L / Nx, L / (2 * Nx)]  # 空間方向の分割幅

# 初期化
x = np.linspace(0, L, Nx + 1)
for dx in dx_values:
    u = initial_condition(x)

    # 計算ループ
    while dt < T:
        u = runge_kutta(u, dt, dx, central_difference)  # または upwind_difference
        dt += dt

    # 結果のプロット
    plt.plot(x, u, label=f'dx = {dx}')

# 結果のプロット
plt.plot(x, u, label=f'dx = {dx}')
plt.title('1D Linear Advection Equation')
plt.xlabel('x')
plt.ylabel('u(x, t=1)')
plt.legend()

# 画像を保存
plt.savefig('result_plot.png')

# 画像を表示
plt.show()

