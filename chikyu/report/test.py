import numpy as np

x = np.arange(1, 10, dtype=float)
x = x.reshape(3, 3)
print(x)

b = 1


def test(a, b):
    a[0, :] = 0
    b = 2
    return a, b


b = test(x, b)[1]
print(x, b)
