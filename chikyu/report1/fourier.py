import numpy as np
import matplotlib.pyplot as plt


def fourier1(N, n):
    T = np.linspace(0, 2*np.pi, n)
    Y = np.zeros(n, dtype=float)
    for k in range(-N, N+1):
        if k == 0:
            pass
        elif k % 2 == 0:
            pass
        else:
            Y += (-4/((np.pi**2)*(k**2)))*np.real(np.exp(1j*k*T))
    return T, Y


def f1(n):
    T = np.linspace(0, 2*np.pi, n)
    Y = np.zeros(n, dtype=float)
    for i in range(n):
        if T[i] < np.pi:
            Y[i] = -1 + 2*T[i]/np.pi
        else:
            Y[i] = 3 - 2*T[i]/np.pi
    return T, Y


memo = {}
t, y = f1(500)
for i in range(1, 5):
    memo[2*i], memo[2*i+1] = fourier1(2*i, 500)
    # plt.plot(t, y, label='f(t)')
    # plt.plot(memo[2*i], memo[2*i+1], label='n = {}'.format(2*i))
    # plt.legend()

for i in range(1,5):
    plt.plot(t, memo[2*i+1]-y, label='difference when n = {}'.format(2*i))
    plt.legend()

plt.show()


def fourier2(N, n):
    T = np.linspace(0, 2*np.pi, n)
    Y = np.zeros(n, dtype=float)
    for k in range(-N, N+1):
        if k == 0:
            pass
        elif k % 2 == 0:
            pass
        else:
            Y += np.real(-2j/(np.pi*k)*np.exp(1j*k*T))
    return T, Y


def f2(n):
    T = np.linspace(0, 2*np.pi, n)
    Y = np.zeros(n, dtype=float)
    for i in range(n):
        if T[i] < np.pi:
            Y[i] = 1
        else:
            Y[i] = -1
    return T, Y


memo1 = {}
t, y = f2(1000)
for i in range(1, 5):
    memo1[2*i], memo1[2*i+1] = fourier2(5*i, 1000)
    # plt.figure()
    # plt.plot(t, y, label='f(t)')
    # plt.plot(memo1[2*i], memo1[2*i+1], label='n = {}'.format(3*i))
    # plt.legend()

plt.figure()
for i in range(1,5):
    plt.plot(t, memo1[2*i+1]-y, label='difference when n = {}'.format(3*i))
    plt.legend()

plt.show()
