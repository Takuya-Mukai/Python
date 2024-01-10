import matplotlib.pyplot as plt
import numpy as np

a = np.array([[1, -1, 0, 0],
              [-1, 2, -1, 0],
              [0, -1, 2, -1],
              [0, 0, -1, 2]])
b = np.array([1, 0, 0, 0])


# SOR method
def sor(a, b, w, dx):
    n = len(b)
    a = np.array(a, dtype='float64')
    b = np.array(b, dtype='float64')

    # fix initial x for test
    x = np.array([0, 0, 0, 0], dtype='float64')

    # copy x in order to compare the difference
    copy_x = np.copy(x)
    difference = 1
    dif = []
    while difference > dx:
        for i in range(n):
            x_childa = (b[i] - a[i, :i-1]@x[:i-1] -
                        a[i, i+1:]@x[i+1:]) / a[i, i]
            x[i] = (1-w)*x[i] + w*x_childa

        difference = np.linalg.norm(x - copy_x)
        dif.append(difference)

        # copy x in order to compare the difference
        copy_x = np.copy(x)
    return x, dif


dx = 1e-3

for i in range(1, 10):
    w = i/10
    x, dif = sor(a, b, w, dx)
    plt.plot(dif, label="omega={}".format(w))

plt.legend()
plt.show()
print(x)
