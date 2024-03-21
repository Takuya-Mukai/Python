import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# %% Importing the dataset
a = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL', a)
# %% load data
df = pd.read_csv(a, header=None, encoding='utf-8')
df.tail()

# %% extract data for the first 100 class labels
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
x = df.iloc[0:100, [0, 2]].values

# %%
# plot data of setosa
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
# plot data of versicolor
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')

# %%
plt.show()
# %%
