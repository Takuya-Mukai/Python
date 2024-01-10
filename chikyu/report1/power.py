import numpy as np
import matplotlib.pyplot as plt


# define funtion and descrete it  by given time
def f(t):
    Y = np.zeros(len(t))
    for i in range(len(t)):
        Y[i] = np.sin(t[i]) + np.sin(2**0.5*t[i])

    return Y



# calculate the power spectrum
def F(f, T):
    N = len(f)
    w = np.zeros(int(N))
    for i in range(int(N)):
        w[i] = i*(2*np.pi)/T

    # calculate the fourier transform
    Fk = np.zeros(int(N), dtype = complex)
    for k in range(len(Fk)):
        Fk[k] = (T/N)*sum(f[n]*np.exp(-1j * (w[k]*T/N) * n)
                          for n in range(len(f)))


    # calculate the power spectrum from the fourier transform
    Pt = np.zeros(len(Fk))
    for i in range(len(Pt)):
        Pt[i] = (np.abs(Fk[i])**2)/T
    return w, Pt


T1 = 50*np.pi
N1 = 1000

# descrete time
t = np.zeros(N1)
for i in range(N1):
    t[i] = i*T1/N1

y = f(t)
w, Pt = F(y, T1)
# Parseval's identity
sum1 = sum(y**2)*T1/N1
sum2 = sum(Pt)
print(sum1, sum2)

plt.plot(w, Pt)
plt.show()
