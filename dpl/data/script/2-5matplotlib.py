import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# np.random.seed(0)
# x = np.random.randn(30)
# y = np.sin(x) + np.random.randn(30)
# plt.figure(figsize = (20, 6))
# plt.plot(x, y, 'o')
# plt.title('title')
# plt.xlabel('x')
# plt.ylabel('y')
#
# plt.grid(True)



# plt.figure(figsize = (20, 6))
# plt.subplot(2, 1, 1)
# x = np.linspace(0, 10, 100)
# plt.plot(x, np.sin(x))
# plt.subplot(2, 1, 2)
# y = np.linspace(0, 10, 100)
# plt.plot(y, np.sin(2*y))
# plt.grid(True)

plt.figure()
plt.subplot(2, 1, 1)
x1 = np.random.uniform(0.0, 1.0, 1000)
y1 = np.random.uniform(0.0, 1.0, 1000)
x2 = np.random.uniform(0.0, 1.0, 1000)
y2 = np.random.uniform(0.0, 1.0, 1000)

plt.plot(x1, y1, 'o', color = 'black')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'o', color = 'red')
plt.savefig('2-5matplotlib.png')
