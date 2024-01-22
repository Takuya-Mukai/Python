import numpy as np


def df(x, t):
    return (x + (x**2 + t**2)**0.5)/t


def runge_kutta(x, t, df, dt):
    k1h = dt * df(x, t)
    k2h = dt * df(x + k1h/2, t + dt/2)
    k3h = dt * df(x + k2h/2, t + dt/2)
    k4h = dt * df(x + k3h, t + dt)
    x_new = x + (k1h + 2*k2h + 2*k3h + k4h)/6
    return x_new


dt = 0.01
x = 0
t = 1

for i in range(1000):
    x = runge_kutta(x, t, df, dt)
    t = t + dt

print(x)

