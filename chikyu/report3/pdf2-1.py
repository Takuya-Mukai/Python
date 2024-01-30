#calculate vection field by runge-kutta method
import numpy as np
import matplotlib.pyplot as plt


def upwind(u):
    du = np.zeros(len(u))
    for i in range(0, len(u)):
        if i == 0:
            du[0] = -(u[0] - u[len(u)-2])/dx
        else:
            du[i] = -(u[i] - u[i-1])/dx

    return du

def central(u):
    du = np.zeros(len(u))
    for i in range(len(u)):
        if i ==0:
            du[0] = -(u[1] - u[-2])/(2*dx)
        elif i ==len(u)-1:
            du[i] = -(u[1] - u[i-1])/(2*dx)
        else:
            du[i] = -(u[i+1] - u[i-1])/(2*dx)
    return du

def runge_kutta(u, scheme, dt):
    u_new = np.copy(u)
    k1 = dt*scheme(u_new)
    k2 = dt*scheme(u_new + k1/2)
    k3 = dt*scheme(u_new + k2/2)
    k4 = dt*scheme(u_new + k3)
    u_new += (k1 + 2*k2 + 2*k3 + k4)/6
    return u_new



x = np.linspace(0, 1, 101)
dx = 0.01
dt = 0.01
u = np.zeros(101)
for i in range(100):
    u[i] = np.sin(np.pi*x[i])**100
plt.figure()
plt.plot(x, u)

for i in range(100):
    u = runge_kutta(u, central, dt)
plt.plot(x, u)
plt.savefig("pde2-1-central.png")

x = np.linspace(0, 1, 101)
dx = 0.01
dt = 0.01
u = np.zeros(101)
for i in range(100):
    u[i] = np.sin(np.pi*x[i])**100
plt.figure()
plt.plot(x, u)

for i in range(100):
    u = runge_kutta(u, upwind, dt)
plt.plot(x, u)
plt.savefig("pde2-1-upwind.png")
