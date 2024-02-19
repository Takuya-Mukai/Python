import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sklearn

import requests, zipfile
import io

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
res = requests.get(url).content
auto = pd.read_csv(io.StringIO(res.decode("utf-8")), header=None)

auto.columns = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

auto = auto[["price", "horsepower", "width", "height"]]
auto.isin(["?"]).sum()
auto = auto.replace("?", np.nan).dropna()
auto = auto.assign(price=pd.to_numeric(auto.price))
auto = auto.assign(horsepower=pd.to_numeric(auto.horsepower))
# print(auto.corr())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# designate the responsive variable as price,
# and explanatory variables as others

X = auto.drop("price", axis=1)
y = auto["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# initialize the model and fit of multiple linear regression

model = LinearRegression()
model.fit(X_train, y_train)

print("coefficients(train):{:.3f}".format(model.score(X_train, y_train)))
print("coefficients(test):{:.3f}".format(model.score(X_test, y_test)))

# print regression coefficients and intercept
print("\nregrettion coefficients\n{}".format(pd.Series(model.coef_, index=X.columns)))
print("\nintercept: {:.3f}".format(model.intercept_))
