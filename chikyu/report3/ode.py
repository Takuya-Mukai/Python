import numpy as np
import matplotlib.pyplot as plt


# def u(x):
#     if x < 1:
#         return np.sin(np.pi*x)**100
#     if x >= 1:
#         return u(x-1)


def du(u):
    du_new = np.zeros(len(u))
    for i in range(len(u)):
        if i == 0:
            du_new[i] = (u[i] - u[len(u)-2])/dx
        else:
            du_new[i] = (u[i] - u[i-1])/dx
    return du_new

    # return (u[0:len(u)] - u[2:len(u)+2])/dx


def runge_kutta(u):
    k1 = dt*du(u)
    k2 = dt*du(u + 0.5 * k1)
    k3 = dt*du(u + 0.5 * k2)
    k4 = dt*du(u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


dx = 0.01
dt = 0.005
x = np.linspace(0, 1, 101)

u = np.sin(np.pi*x)**100
plt.plot(x, u)
plt.show()
for i in range(20):
    u = runge_kutta(u)

