import numpy as np
import matplotlib.pyplot as plt


def du_upwind(u):
    du_new = np.zeros(len(u))
    for i in range(len(u)):
        if i == 0:
            du_new[i] = -(u[len(u)-1] - u[len(u)-2])/dx
        else:
            du_new[i] = -(u[i] - u[i-1])/dx
    return du_new


def runge_kutta(u, du):
    k1 = dt*du(u)
    k2 = dt*du(u + 0.5 * k1)
    k3 = dt*du(u + 0.5 * k2)
    k4 = dt*du(u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


dx = 0.01
dt = 0.01
x = np.linspace(0, 1, 101)

u = np.sin(np.pi*x)**100
plt.plot(x, u)
for i in range(1,100):
    u = runge_kutta(u, du_upwind)
    # if i % 5 == 0:
    #     plt.plot(x, u)

plt.plot(x, u)

plt.savefig('ode.png')
