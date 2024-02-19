import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import linear_model
import requests, zipfile
from io import StringIO
import io

student_data_math = pd.read_csv('student-mat.csv', sep=';')
student_data_math.info()

# plt.boxplot(student_data_math['G1'])
# plt.grid(True)

# print(student_data_math['absences'].std(ddof = 0) / student_data_math['absences'].mean())

print(student_data_math.describe())
print(student_data_math["G1"])
print(type(student_data_math["G1"]))

plt.plot(student_data_math['G1'], student_data_math['G3'], 'o')
plt.ylabel('G3 grade')
plt.xlabel('G1 grade')
plt.grid(True)
plt.savefig('3-1-2.png')
