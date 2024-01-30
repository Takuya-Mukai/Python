import numpy as np
import matplotlib.pyplot as plt
def dx(x, y, u, v):
    return u
def dy(x, y, u, v):
    return v
def du(x, y, u, v):
    return -x/(x**2+y**2)**(3/2)
def dv(x, y, u, v):
    return -y/(x**2+y**2)**(3/2)

dt = 0.01
def runge_kutta(x, y, u, v, dx, dy, du, dv, dt):
    k1x = dx(x, y, u, v)*dt
    k1y = dy(x, y, u, v)*dt
    k1u = du(x, y, u, v)*dt
    k1v = dv(x, y, u, v)*dt

    k2x = dx(x+k1x/2, y+k1y/2, u+k1u/2, v+k1v/2)*dt
    k2y = dy(x+k1x/2, y+k1y/2, u+k1u/2, v+k1v/2)*dt
    k2u = du(x+k1x/2, y+k1y/2, u+k1u/2, v+k1v/2)*dt
    k2v = dv(x+k1x/2, y+k1y/2, u+k1u/2, v+k1v/2)*dt
    
    k3x = dx(x+k2x/2, y+k2y/2, u+k2u/2, v+k2v/2)*dt
    k3y = dy(x+k2x/2, y+k2y/2, u+k2u/2, v+k2v/2)*dt
    k3u = du(x+k2x/2, y+k2y/2, u+k2u/2, v+k2v/2)*dt
    k3v = dv(x+k2x/2, y+k2y/2, u+k2u/2, v+k2v/2)*dt

    k4x = dx(x+k3x, y+k3y, u+k3u, v+k3v)*dt
    k4y = dy(x+k3x, y+k3y, u+k3u, v+k3v)*dt
    k4u = du(x+k3x, y+k3y, u+k3u, v+k3v)*dt
    k4v = dv(x+k3x, y+k3y, u+k3u, v+k3v)*dt

    x += (k1x+2*k2x+2*k3x+k4x)/6
    y += (k1y+2*k2y+2*k3y+k4y)/6
    u += (k1u+2*k2u+2*k3u+k4u)/6
    v += (k1v+2*k2v+2*k3v+k4v)/6

    return x, y, u, v


x = np.zeros(1000)
y = np.zeros(1000)
u = np.zeros(1000)
v = np.zeros(1000)
dt = 0.01
x[0] = 3
y[0] = 0
u[0] = 0.3
v[0] = 0.2
for i in range(1000-1):
    x[i+1], y[i+1], u[i+1], v[i+1] = runge_kutta(x[i], y[i], u[i], v[i], dx, dy, du, dv, dt)

plt.figure()
plt.plot(x, y)
plt.savefig('ode-2.png')
