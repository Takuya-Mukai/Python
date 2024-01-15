import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
dx = 0.01


def u(x, t):
  if x >= 1:
    return u(x-1, t)
  elif t ==0:
    return (np.sin(2*np.pi*x))**2
  else:


def runge_kutta(x, t, h):
k1 = h * f(x,t)
k2 = h * f(x + k1/2, t + h/2)
k3 = h * f(x + k2/2, t + h/2)
k4 = h * f(x + k3, t + h)
return x + (k1 + 2*k2 + 2*k3 + k4)/6

