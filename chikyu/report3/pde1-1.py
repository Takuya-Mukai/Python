#laplace equation by jacobi method
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def boundary_condition(x, y):
    if y == 0:
        return x
    elif y == 1:
        return 1-x
    elif x == 0:
        return y
    elif x == 1:
        return 1-y
    else:
        return 0

x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
u = np.zeros((101, 101), dtype = float)

for i in range(101):
    for j in range(101):
        u[i, j] = boundary_condition(x[i], y[j])

diff = 1.0
trial = 0
while diff > 1e-6:
    u_new = u.copy()
    trial +=1

    u_new[1:-1, 1:-1] = (u[1:-1, 2:] + u[1:-1, 0:-2] + u[2:, 1:-1] + u[0:-2, 1:-1])/4
    # for i in range(1,100):
    #     for j in range(1,100):
    #         u_new[i,j] = 0.25*(u[i,j-1]+u[i,j+1]+u[i-1,j]+u[i+1,j])

    diff = min(diff, np.abs(u-u_new).max())
    print(trial, diff, np.abs(u-u_new).max())
    u = u_new


fig = plt.figure()
plt.pcolor(x, y, u)
plt.colorbar()
plt.show()
plt.savefig('pde1-1.png')
