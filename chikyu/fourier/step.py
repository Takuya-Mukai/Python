import numpy as np
import matplotlib.pyplot as plt


def step(n, L):
    t = np.linspace(0, L, 1000)
    y = sum([(np.pi*(2*i - 1))*np.sin((2*np.pi*(2*i - 1)*t)/L)
            for i in range(1, n+1)])
    plt.plot(t, y)
    plt.show()


step(10, 1)
