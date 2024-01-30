import numpy as np
import matplotlib.pyplot as plt
import scipy

def u(x, y):
    if y == 0:
        return x
    elif y == 1:
        return 1-x
    elif x == 0:
        return y
    elif x == 1:
        return 1-y

x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)


