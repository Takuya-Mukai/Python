import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

hier_df = DataFrame(
    np.arange(9).reshape((3, 3)),
    index=[["a", "a", "b"], [1, 2, 2]],
    columns=[["Osaka", "Tokyo", "Osaka"], ["Blue", "Red", "Red"]],
)

# # name to index
# hier_df.index.names = ["key1", "key2"]
# print(hier_df)
#
# print(hier_df["Osaka"])

data1 = {
    "id": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
    "city": [
        "Tokyo",
        "Osaka",
        "Kyoto",
        "Hokkaido",
        "Tokyo",
        "Tokyo",
        "Osaka",
        "Kyoto",
        "Hokkaido",
        "Tokyo",
    ],
    "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
    "name": [
        "Hiroshi",
        "Takashi",
        "Hayato",
        "Satoru",
        "Steve",
        "Mituru",
        "Aou",
        "Tarou",
        "Suguru",
        "Mitsuo",
    ],
}
df1 = DataFrame(data1)
print(df1)
data2 = {
    "id": ["100", "101", "102", "105", "107"],
    "math": [50, 43, 33, 76, 98],
    "english": [90, 30, 20, 50, 30],
    "sex": ["M", "F", "F", "M", "M"],
    "index_num": [0, 1, 2, 3, 4],
}
df2 = DataFrame(data2)


df1["up_two_num"] = df1["birth_year"].map(lambda x: str(x)[0:3])

print(df1.groupby("city", as_index=0))

math = pd.read_csv("student-mat.csv", sep=";")

# print(math.groupby(["school", "sex"])["G1"].mean())

# functions = ['count', 'mean', 'max', 'min']
# grouped_math = math.groupby(['sex', 'address'])
# print(grouped_math[['age', 'G1']].agg(functions))

# chapter 6-4
