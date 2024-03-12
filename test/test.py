import numpy as np

x = np.arange(9)
x = x.reshape(3, 3)

it = np.nditer(x, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index
    print(idx, x[idx], type(idx))
    it.iternext()
