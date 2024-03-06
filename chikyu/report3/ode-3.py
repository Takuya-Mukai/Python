import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -x

x = np.zeros(1000)
t = np.linspace(0,10,1000)
h = 0.01

x[0] = np.exp(h)
for i in range(1, 999):
    x[i+1] = x[i-1] + 2*h*f(x[i])

plt.figure()
plt.plot(t, x)
plt.savefig('ode-3.png')
