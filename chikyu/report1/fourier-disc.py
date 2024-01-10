import numpy as np
import matplotlib.pyplot as plt

# function for calculate c-hat from the given f which is discreted
def fourier_disc(f):
    N = len(f)
    c_hat = np.zeros(N, dtype=complex)

    for k in range(N):
        c_hat[k] = sum(f[n] * (np.exp(-1j * 2*np.pi * n * k / N))
                       for n in range(N))/N
    return c_hat

# function to reproduce f from the given c-hat
def fourier_rest(c_hat):
    N = len(c_hat)
    fn = np.zeros(N, dtype=float)

    for n in range(N):
        fn[n] = np.real(sum(c_hat[k] * (np.exp(1j * 2*np.pi * k * n / N))
                    for k in range(N)))
    return fn

# function to decide the value of f(t)
def f(t):
    if 0 <= t < np.pi:
        return -1 + 2*t/np.pi
    if np.pi <= t < 2*np.pi:
        return 3 - 2*t/np.pi

# function for examin c-hat work well or not
def f1(t):
  return np.sin(t) + np.sin(2**0.5*t)

# function to descrete the f(t)
def fn(n):
    fn = np.zeros(n)
    for i in range(n):
        fn[i] = f(i * 2*np.pi/n)
    return fn

# function to descrete the f(t) for examin
def fn1(n):
    fn = np.zeros(n)
    for i in range(n):
        # this is examin for f(t) = sin(t) + sin(sqrt(2)*t)
        fn[i] = f1(i * 2*np.pi/n)
    return fn


# function for calculate c directly from the given f
def fourier1(n):
    c = np.zeros(n, dtype=float)
    for k in range(n):
        if k == 0:
            c[k] = 0
        elif k % 2 == 0:
            c[k] = 0
        else:
            c[k] = (-4/((np.pi**2)*(k**2)))
    return c


n = 100

k = np.zeros(n)
for i in range(n):
  k[i] = i * 2*np.pi/n

# discrete f
fn = fn(n)
#discrete f for examin
fn1 = fn1(n)

# calculate c-hat from discreted fn
c_hat = fourier_disc(fn)
c_hat1 = fourier_disc(fn1)

# reproduce fn from c-hat
fn_dash1 = fourier_rest(c_hat1)

# plot reproduced f(t) and original f(t)
plt.plot(k, fn_dash1, label = 'reproduced')
plt.plot(k, fn1, "--", label = 'original')
plt.legend()
plt.show()

plt.figure()
# calculate directry c from f 
c = fourier1(n)
# plot the difference between c and c-hat
plt.plot(k, c-c_hat)
plt.show()
