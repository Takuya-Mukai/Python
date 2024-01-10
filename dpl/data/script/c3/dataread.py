import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

student_data_math = pd.read_csv('student-mat.csv', sep=';')
# print(student_data_math.head())
# print(student_data_math.info())
# print(student_data_math.groupby('sex')['age'].mean())

