import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

por = pd.read_csv('student-por.csv', sep=';')
math = pd.read_csv('student-mat.csv', sep=';')
X = por.loc[:,['G1']].values
Y = por['G3'].values
print(X,Y)
reg = linear_model.LinearRegression()

reg.fit(X, Y)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
print('coefficients of determination:', reg.score(X, Y))
plt.scatter(X, Y)
plt.plot(X, reg.predict(X))
plt.grid()
plt.savefig('3-test.png')
plt.figure()
Y1 = por.loc['absences']
reg.git(X, Y1)
print('Coefficients: \n', reg.coef_)
print('intercept: \n', reg.intercept_)
print('coefficients of determination:', reg.score(X, Y1))
plt.scatter(X, Y1)
plt.plot(X, reg.predict(X))
plt.grid()
plt.savefig('3-test1.png')
plt.figure()

