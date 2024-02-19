import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

student_data_math = pd.read_csv('student-mat.csv', sep=';')
#make instance of linear regression
reg = linear_model.LinearRegression()

X = student_data_math.loc[:, ['G1']].values
Y = student_data_math['G3'].values
reg.fit(X, Y)
print('regression coefficient:', reg.coef_)
# plt.plot(student_data_math['G1'], student_data_math['G3'], 'o')
# plt.grid()

print('intercept:', reg.intercept_)
plt.scatter(X, Y)
plt.xlabel('G1 grade')
plt.ylabel('G3 grade')
plt.plot(X, reg.predict(X))
plt.grid(True)
print('coefficients of determination:', reg.score(X, Y))
plt.savefig('3-4.png')
