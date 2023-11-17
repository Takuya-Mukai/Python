#!/usr/bin/env python
# coding: utf-8

# # Chapter 5 練習と総合問題解答

# In[ ]:


# 以下のライブラリを使うので、あらかじめ読み込んでおいてください
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series,DataFrame
import pandas as pd

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline

# 小数第３まで表示
%precision 3

# #### <練習問題 5-1>
# 
# 以下に示す`sample_names`と`data`という2つの配列があるとします。ブールインデックス参照をつかって、`data`から、`sample_names`の`b`に該当するデータを抽出してください。

# In[ ]:


# データの準備
sample_names = np.array(['a','b','c','d','a'])
random.seed(0)
data = random.randn(5,5)

print(sample_names)
print(data)

# In[ ]:


# 解答
data[sample_names == 'b']

# #### <練習問題 5-2>
# 
# <練習問題 5-1>で使ったデータ`sample_names`と`data`を使って、`data`から、`sample_names`の`c`以外に該当するデータを抽出してください。
# 

# In[ ]:


# 解答
data[sample_names != 'c']

# #### <練習問題 5-3>
# 
# 次の`x_array`、`y_array`があるとき、Numpyの`where`を用いて条件制御し、3番目と4番目は`x_array`から、1番目、2番目、5番目は`y_array`から、それぞれ値を取り出したデータを生成してください。

# In[ ]:


x_array= np.array([1,2,3,4,5])
y_array= np.array([6,7,8,9,10])

# In[ ]:


# 解答
cond_data = np.array([False,False,True,True,False])
# 条件制御実施
print(np.where(cond_data,x_array,y_array))

# #### <練習問題 5-4>
# 
# 以下のデータに対して、すべての要素の平方根を計算した行列を表示してください。

# In[ ]:


sample_multi_array_data2 = np.arange(16).reshape(4,4)
sample_multi_array_data2 

# In[ ]:


# 解答
np.sqrt(sample_multi_array_data2)

# #### <練習問題 5-5>
# 
# <練習問題 5-4>のデータ`sample_multi_array_data2`の最大値、最小値、合計値、平均値を求めてください。

# In[ ]:


# 解答
print('最大値:',sample_multi_array_data2.max())
print('最小値:',sample_multi_array_data2.min())
print('合計値:',sample_multi_array_data2.sum())
print('平均値:',sample_multi_array_data2.mean())

# #### <練習問題 5-6>
# 
# <練習問題 5-4>のデータ`sample_multi_array_data2`の対角成分の和を求めてください。

# In[ ]:


# 解答
print('対角成分の和:',np.trace(sample_multi_array_data2))

# #### <練習問題 5-7>
# 
# 次の2つの配列に対して、縦に結合してみましょう。

# In[ ]:


# データの準備
sample_array1 = np.arange(12).reshape(3,4)
sample_array2 = np.arange(12).reshape(3,4)

# In[ ]:


# 解答
np.concatenate([sample_array1,sample_array2])

# #### <練習問題 5-8>
# 
# <練習問題 5-7>の2つの配列に対して、横に結合してみましょう。

# In[ ]:


# 解答
np.concatenate([sample_array1,sample_array2],axis=1)

# #### <練習問題 5-9>
# 
# Pythonにおけるリストの各要素に3を加えるためにはどうすればよいでしょうか。numpyのブロードキャスト機能を使ってください。

# In[ ]:


sample_list = [1,2,3,4,5]

# In[ ]:


# 解答
np.array(sample_list)+3

# #### <練習問題 5-10>
# 
# 以下のデータに対して、線形補間の計算をして、グラフを描いてください。

# In[ ]:


x = np.linspace(0, 10, num=11, endpoint=True)
y = np.sin(x**2/5.0)
plt.plot(x,y,'o')
plt.grid(True)

# In[ ]:


# 解答
from scipy import interpolate

# 線形補間
f = interpolate.interp1d(x, y,'linear')
plt.plot(x,f(x),'-')
plt.grid(True)

# #### <練習問題 5-11>
# 
# 2次元のスプライン補間（点と点の間を2次の多項式で補間する方法）を使って<練習問題 5-10>のグラフに書き込んでください（2次元のスプライン補間はパラメータを`quadratic`とします）。

# In[ ]:


# スプライン2次補間も加えて、まとめてみる、
f2 = interpolate.interp1d(x, y,'quadratic')

#曲線を出すために、xの値を細かくする。
xnew = np.linspace(0, 10, num=30, endpoint=True)

# グラフ化
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')

# 凡例
plt.legend(['data', 'linear', 'quadratic'], loc='best')
plt.grid(True)

# #### <練習問題 5-12>
# 
# 3次元のスプライン補間も加えてみましょう。

# In[ ]:


# 解答
# スプライン2,3次補間も加えて、まとめてみる、
f2 = interpolate.interp1d(x, y,'quadratic')
f3 = interpolate.interp1d(x, y,'cubic')

#曲線を出すために、xの値を細かくする。
xnew = np.linspace(0, 10, num=30, endpoint=True)

# グラフ化
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '--')

# 凡例
plt.legend(['data', 'linear','quadratic','cubic'], loc='best')
plt.grid(True)

# #### <練習問題 5-13>
# 以下の行列に対して、特異値分解をしてください。

# In[ ]:


B = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
B

# In[ ]:


# 解答
# 特異値分解の関数linalg.svd
U, s, Vs = sp.linalg.svd(B)
m, n = B.shape

S = sp.linalg.diagsvd(s,m,n)

print('U.S.V* = \n',U@S@Vs)

# #### <練習問題 5-14>
# 以下の行列に対して、LU分解をして、$Ax=b$の方程式を解いてください。

# In[ ]:


#データの準備
A = np.identity(3)
print(A)
A[0,:] = 1
A[:,0] = 1
A[0,0] = 3
b = np.ones(3)
print(A)
print(b)

# In[ ]:


# 解答
# 正方行列をLU分解する
(LU,piv) = sp.linalg.lu_factor(A)

L = np.identity(3) + np.tril(LU,-1)
U = np.triu(LU)
P = np.identity(3)[piv]

# 解を求める
sp.linalg.lu_solve((LU,piv),b)

# In[ ]:


# 確認
np.dot(A,sp.linalg.lu_solve((LU,piv),b))

# #### <練習問題 5-15>
# 以下の積分を求めてみましょう。

# \begin{eqnarray}
#   \int_0^2 (x+1)^2 dx
# \end{eqnarray}

# In[ ]:


# 解答
from scipy import integrate

def calc1(x):
    return (x+1)**2

# 計算結果と推定誤差
integrate.quad(calc1, 0, 2)

# #### <練習問題 5-16>
# cos関数の範囲$(0,\pi)$の積分を求めてみましょう。

# In[ ]:


# 解答
import math
from numpy import cos

integrate.quad(cos, 0, math.pi/1)

# #### <練習問題 5-17>
# Sicpyを用いて、以下の関数が0となる解を求めましょう。

# \begin{eqnarray*}
# \ f(x) = 5x -10
# \end{eqnarray*}

# In[ ]:


# 解答
def f(x):
    y =  5*x - 10
    return y

# In[ ]:


# 解答
x = np.linspace(0,4)
plt.plot(x,f(x))
plt.plot(x,np.zeros(len(x)))
plt.grid(True)

# In[ ]:


# 解答
from scipy.optimize import fsolve

x = fsolve(f,2)
print(x)

# #### <練習問題 5-18>
# 同様に、以下の関数が0となる解を求めましょう。

# \begin{eqnarray*}
# \ f(x) = x^3 - 2x^2 - 11x +12
# \end{eqnarray*}

# In[ ]:


# 解答
def f2(x):
    y =  x**3 - 2 * x**2 - 11 * x + 12
    return y

# In[ ]:


# 解答
x = np.linspace(-5,5)
plt.plot(x,f2(x))
plt.plot(x,np.zeros(len(x)))
plt.grid(True)

# グラフから解は-3と1と4付近にあることがわかります。

# In[ ]:


# 解答
from scipy.optimize import fsolve

x = fsolve(f2,-3)
print(x)

# In[ ]:


# 解答
# x = 1 付近
x = fsolve(f2,1)
print(x)

# In[ ]:


# 解答
# x = 4 付近
x = fsolve(f2,4)
print(x)

# ## 5.4 総合問題

# ### ■総合問題5-1 コレスキー分解

# 以下の行列に対して、コレスキー分解をして、$Ax=b$の方程式を解いてください。

# In[ ]:


A = np.array([[5, 1, 0, 1],
              [1, 9, -5, 7],
              [0, -5, 8, -3],
              [1, 7, -3, 10]])
b = np.array([2, 10, 5, 10])

# In[ ]:


# 解答
L = sp.linalg.cholesky(A)

t = sp.linalg.solve(L.T.conj(), b)
x = sp.linalg.solve(L, t)

print(x)

# In[ ]:


# 確認
np.dot(A,x)

# numpyを使っても計算できます。

# In[ ]:


# 解答
L = np.linalg.cholesky(A)

t = np.linalg.solve(L, b)
x = np.linalg.solve(L.T.conj(), t)

print(x)

# In[ ]:


# 確認
np.dot(A,x)

# ### ■総合問題5-2 積分

# $0≤ x ≤ 1$、$0≤y≤1−x$ の三角領域で定義される以下の関数の積分値を求めてみましょう。
# \begin{eqnarray}
#   \int_0^1 \int_0^{1-x} 1/(\sqrt{(x+y)}(1+x+y)^2) dy dx 
# \end{eqnarray}
# 

# In[ ]:


from scipy import integrate
import math

# 解答
integrate.dblquad(lambda x, y: 1/(np.sqrt(x+y)*(1+x+y)**2), 0, 1, lambda x: 0, lambda x: 1-x)

# ### ■総合問題5-3 最適化問題

# 以下の最適化問題をSicpyを使って解いてみましょう。

# \begin{eqnarray*}
# \ min \ f(x) = x^2+1
# \\  s.t. x \ge -1
# \end{eqnarray*}

# In[ ]:


# 解答
from scipy.optimize import minimize

# 目的関数
def func(x):
    return x ** 2 + 1

# 制約条件式
def cons(x):
    return (x + 1)

cons = (
    {'type': 'ineq', 'fun': cons}
)
x = -10 # 初期値は適当

result = minimize(func, x0=x, constraints=cons, method='SLSQP')
print(result)

# In[ ]:


print('Y:',result.fun)
print('X:',result.x)
