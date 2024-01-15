import numpyas np
import matplotlib.pyplot as plt

def f(t, x):
  return (x + (x*2 + t*2))^0.5/t


def runge_kuttta(f, t0, x0, h, n):
  t = t0
  x = y0
  for i in range(100):
    k1 = f(t, x)
    k2 = f(t + h/2, x + k1/2)
    k3 = f(t + h/2, x + k2/2)
    k4 = f(t + h, x + k3)
    x = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
    t = t + h
  return x

x = np.linespace(0, 1, 100)
