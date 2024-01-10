import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
x = 1
v = 0
xl = []
vl = []
tmax = 100
# Euler method

t = 0
while t < tmax:
    x = x + v * dt
    v = v - x * dt
    xl.append(x)
    vl.append(v)
    t = t + dt

plt.plot(xl, vl)

x = 1
v = 0
xl = []
vl = []
tmax = 100
# simplectic Euler method
t = 0
while t < tmax:
    v = v - x * dt
    x = x + v * dt
    xl.append(x)
    vl.append(v)
    t = t + dt

plt.plot(xl, vl)
plt.show()
