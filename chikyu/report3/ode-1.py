import numpy as np
import matplotlib.pyplot as plt

def f(x, t):
    return (x + (x**2 + t**2)**0.5)/t

def runge_kutta(x, t, scheme, dt):
    k1 = dt * scheme(x, t)
    k2 = dt * scheme(x + k1/2, t + dt/2)
    k3 = dt * scheme(x + k2/2, t + dt/2)
    k4 = dt * scheme(x + k3, t + dt)
    return x + (k1 + 2*k2 + 2*k3 + k4)/6



print("-------ode_1-------")
l = [1, 5, 10, 100]
for i in range(len(l)):
    dt = 0.0001*l[i]
    x = 0
    t = 1
    for i in range(int(10/dt)):
        t += dt
        x = runge_kutta(x, t, f, dt)
    print("dt = ",dt, "x = ",x)
