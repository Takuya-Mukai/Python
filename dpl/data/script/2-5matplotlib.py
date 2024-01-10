import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import math
# random.seed(0)
#
# # data of x axis
# x = np.random.randn(30)
#
# # data of y axis
# y = np.sin(x) + np.random.randn(30)
#
# # size of graph
# plt.figure(figsize=(20, 6))
#
# plt.plot(x, y, 'o')
#
# plt.scatter(x, y)
#
# plt.title('Title Name')
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
#
# plt.show()


# np.random.seed(0)
# # range of data
# numpy_data_x = np.arange(1000)
#
# # occurrence and piling of random data
# numpy_random_data_y = np.random.randn(1000).cumsum()
#
# # size of graph
# plt.figure(figsize=(20, 6))
#
# # we can use label by label= and legend
# plt.plot(numpy_data_x, numpy_random_data_y, label='Label')
# plt.leg end()
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()


# # split graph
#
# # define graph size
# plt.figure(figsize=(20, 6))
#
# # 2x1 graph, 1st graph
# plt.subplot(2, 1, 1)
# x = np.linspace(-10, 10, 100)
# plt.plot(x, np.sin(x))
#
# # 2x1 graph, 2nd graph
# plt.subplot(2, 1, 2)
# y = np.linspace(-10, 10, 100)
# plt.plot(y, np.sin(2*y))
#
# plt.grid(True)
#
# plt.show()

# random.seed(0)
# plt.figure(figsize=(20, 6))
# plt.hist(np.random.randn(10 ** 5) * 10 + 50, bins = 60, range = (20,80))
# plt.grid(True)
# plt.show()

#
# np.random.seed(0)
# plt.subplot(2, 1, 1)
# plt.hist(np.random.uniform(0.0, 1.0, 10000), bins = 100)
# plt.grid(True)
#
# plt.subplot(2, 1, 2)
# plt.hist(np.random.uniform(0.0, 1.0, 10000), bins = 100)
# plt.grid(True)
# plt.show()


def montecarlo(n):
    x = np.random.uniform(0.0, 1.0, n)
    y = np.random.uniform(0.0, 1.0, n)
    print(4*sum([1 for i in range(n) if math.hypot(x[i], y[i])<= 1])/(n))


montecarlo(int(input()))
