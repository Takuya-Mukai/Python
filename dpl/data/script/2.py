import numpy as np
import scipy as sp
import numpy.random as random
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

data = np.array([9, 2, 3, 4, 10, 6, 7, 8, 1, 5])
print(data)
print(random.choice(data, 10))
print(random.choice(data, 10, replace=False))
print(random.seed(0))

# practice1
data = np.arange(1, 51)
print(np.sum(data))
data = random.randn(10)
data = 3 * np.ones((5, 5), dtype=np.int64)
print(np.dot(data, data))


attri_data = {'ID': ['1', '2', '3', '4', '5'],
              'Sex': ['F', 'F', 'M', 'M', 'F'],
              'Money': [1000, 2000, 500, 300, 700],
              'Name': ['Saito', 'Horie', 'Kondo', 'Kawada', 'Matsubara']}

attri_data_frame1 = DataFrame(attri_data1)

print(attri_data_frame.groupby('Sex')['Money'].mean())
attri_data2 = {'ID':['3','4','7'],
               'Math': [60, 30, 40],
               'English': [80, 20, 30]}

attri_data_frame2 = DataFrame(attri_data2)


