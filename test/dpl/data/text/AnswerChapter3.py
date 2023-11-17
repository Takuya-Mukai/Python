#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 練習と総合問題解答

# In[ ]:


# 以下のライブラリを使うので、あらかじめ読み込んでおいてください
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
%matplotlib inline

# 小数第３位まで表示
%precision 3

# #### <練習問題 3-1>
# 
# 本章でダウンロードしたポルトガル語の成績データであるstudet-por.csvを読み込んで、要約統計量を表示してください。

# In[ ]:


cd ./chap3

# In[ ]:


# 解答
student_data_por = pd.read_csv('student-por.csv', sep=';')
student_data_por.describe()

# #### <練習問題 3-2>
# 
# 以下の変数をキーとして、数学の成績データ（`student-mat.csv`）とポルトガル語の成績データ（`student-por.csv`）の結果をマージしてください。マージするときは、両方にデータが含まれている（欠けていない）データを対象としてください（内部結合と言います）。
# 
# そして、要約統計量を計算してください。
# 
# なお、以下以外の変数は、それぞれのデータで同名の変数名があり重複するので、`suffixes=('_math', '_por')`のパラメータを追加して、どちらからのデータのものかわかるようにしてください。
# 
# `['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']`

# In[ ]:


student_data_math = pd.read_csv('student-mat.csv', sep=';')

# In[ ]:


# 解答
student_data_merge = pd.merge(student_data_math
                              , student_data_por
                              , on=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu'
                                      , 'Fedu', 'Mjob', 'Fjob', 'reason', 'nursery', 'internet']
                              , suffixes=('_math', '_por'))
student_data_merge.describe()

# In[ ]:


# 補足：同じ変数名だが、データソースが異なるため、同じデータではない例
# 「student_data_merge.traveltime_math」と「student_data_merge.traveltime_por」のデータが同じでない（==）行をカウント
sum(student_data_merge.traveltime_math==student_data_merge.traveltime_por)

# #### <練習問題 3-3>
# 
# <練習問題 3-3>でマージしたデータについて、`Medu`、`Fedu`、`G3_math`などの変数をいくつかピックアップして、散布図とヒストグラムを作成してみましょう。どういった傾向がありますか。また、数学データのみの結果と違いはありますか。考察してみましょう。

# In[ ]:


# 解答
sns.pairplot(student_data_merge[['Medu', 'Fedu', 'G3_math']])
plt.grid(True)

# 考察として、例えば、上のグラフからみるに、MeduやFeduが増えると、G3のスコアもあがるようにみえますが、微妙な差なので、特にこれといった傾向はなさそうです。

# #### <練習問題 3-4>
# 
# ポルトガル語の成績データであるstudent-por.csvのデータを使って、`G3`を目的変数、`G1`を説明変数として単回帰分析を実施し、回帰係数、切片、決定係数を求めてください。

# In[ ]:


student_data_por = pd.read_csv('student-por.csv', sep=';')

# In[ ]:


from sklearn import linear_model

# 線形回帰のインスタンスを生成
reg = linear_model.LinearRegression()

# 説明変数に "一期目の成績" を利用
X = student_data_por.loc[:, ['G1']].values

# 目的変数に "最終の成績" を利用
Y = student_data_por['G3'].values
 
# 予測モデルを計算
reg.fit(X, Y)
 
# 回帰係数
print('回帰係数:', reg.coef_)
 
# 切片 
print('切片:', reg.intercept_)

 # 決定係数、寄与率とも呼ばれる
print('決定係数:', reg.score(X, Y))

# #### <練習問題 3-5>
# 
# 練習問題3-4のデータの実際の散布図と、回帰直線を合わせてグラフ化してください。

# In[ ]:


# 散布図
plt.scatter(X, Y)
plt.xlabel('G1 grade')
plt.ylabel('G3 grade')

# その上に線形回帰直線を引く
plt.plot(X, reg.predict(X))
plt.grid(True)

# #### <練習問題 3-6>
# 
# student-por.csvのデータを使って、`G3`を目的変数、`absences`（欠席数）を説明変数として単回帰分析を実施し、回帰係数、切片、決定係数を求めてください。また、散布図と回帰直線をグラフ化してみましょう。そして、この結果を見て、考察してみましょう。

# In[ ]:


from sklearn import linear_model

# 線形回帰のインスタンスを生成
reg = linear_model.LinearRegression()

# 説明変数に "欠席数" を利用
X = student_data_por.loc[:, ['absences']].values

# 目的変数に "最終の成績" を利用
Y = student_data_por['G3'].values
 
# 予測モデルを計算
reg.fit(X, Y)
 
# 回帰係数
print('回帰係数:', reg.coef_)
 
# 切片 
print('切片:', reg.intercept_)

 # 決定係数、寄与率とも呼ばれる
print('決定係数:', reg.score(X, Y))

# In[ ]:


# 散布図
plt.scatter(X, Y)
plt.xlabel('absences')
plt.ylabel('G3 grade')

# その上に線形回帰直線を引く
plt.plot(X, reg.predict(X))
plt.grid(True)

# グラフから、右下がり（欠席数が増えれば増えるほど、G3の結果が）のようにも見えますが、決定係数がかなり低いので、あくまで参考に見る程度になります。

# ## 3.5 総合問題

# ### ■総合問題3-1　統計の基礎と可視化
# 
# 以下のサイトにあるデータ（ワインの品質）を読み込み、以下の問いに答えてください。
# 
# http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
# 
# （1）要約統計量（平均、最大値、最小値、標準偏差など）を算出してください。
# 

# なお、pandasには、データをアウトプットできるメソッド（`to_csv`）もありますので、余裕があれば、計算した基本統計量の結果をCSVファイルに保存するところまでやってみましょう。

# In[ ]:


# 解答
# まずはデータを読み込み、先頭5行を読み込み
wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
wine.head()

# たとえば、wine_data.csvという名前でファイルを保存したかったら以下のように実行します。

# In[ ]:


file_name = 'wine_data.csv'
wine.to_csv(file_name)

# #### データの説明
# 
# fixed acidity：酒石酸濃度<br>
# volatile acidity：酢酸酸度<br>
# citric acid：クエン酸濃度<br>
# residual sugar：残留糖濃度<br>
# chlorides：塩化物濃度<br>
# free sulfur dioxide：遊離亜硫酸濃度<br>
# total sulfur dioxide：亜硫酸濃度<br>
# density：密度<br>
# pH：pH<br>
# sulphates：硫酸塩濃度<br>
# alcohol：アルコール度数<br>
# quality：0-10 の値で示される品質のスコア<br>

# In[ ]:


# 解答(1)
wine.describe()

# （2）それぞれの変数の分布と、それぞれの変数の関係性（2変数間のみ）がわかるように、グラフ化してみましょう。すべての変数を用いて実行すると時間がかかりますので、注意しましょう。何かわかる傾向はありますか。

# In[ ]:


#解答(2)
sns.pairplot(wine)

# 上の散布図について、相関があるものないものがあるようです。

# ### ■総合問題3-2 ローレンツ曲線とジニ係数
# 
# 本章で使用したstudent_data_mathのデータを使って、以下の問いに答えてください。以下で使用する変数は、「`student_data_math = pd.read_csv('student-mat.csv', sep=';')`」のようにして`student-mat.csvをDataFrame`として読み込んだデータです。 
# 
# 以下で扱うローレンツ曲線やジニ係数は、貧富の格差（地域別、国別など）を見るための指標として使われています。なお、本問題は少し難易度が高いため、参考程度に見てください。ローレンツ曲線やジニ計数について詳しく知りたい方は、参考文献「A-5」などを参照してみてください。
# 
# （1）一期目の数学データ（G1変数）について、男女別に昇順に並び替えをしてください（男=sexが"F"、女=sexが"M"のデータです）。そして、横軸に人数の累積比率、縦軸に一期目の値の累積比率をとってください。この曲線をローレンツ曲線といいます。このローレンツ曲線を男女別に一期目の数学成績でグラフ化してください。なお、累積比率については、同じような計算を第2章のNumpyでやっていますので（積み上げ割合）、それを参考にしてください。

# In[ ]:


# 解答(1)
student_data_math_F = student_data_math[student_data_math.sex=='F']
student_data_math_M = student_data_math[student_data_math.sex=='M']

# 昇順にする
sorted_data_G1_F = student_data_math_F.G1.sort_values()
sorted_data_G1_M = student_data_math_M.G1.sort_values()

# グラフ作成用のデータ
len_F = np.arange(len(sorted_data_G1_F))
len_M = np.arange(len(sorted_data_G1_M))

# ローレンツ曲線
plt.plot(len_F/len_F.max(), len_F/len_F.max(), label='E') # 完全平等
plt.plot(len_F/len_F.max(), sorted_data_G1_F.cumsum()/sorted_data_G1_F.sum(), label='F')
plt.plot(len_M/len_M.max(), sorted_data_G1_M.cumsum()/sorted_data_G1_M.sum(), label='M')
plt.legend()
plt.grid(True)

# 
# 
# （2）不平等の程度を数値で表したものをジニ係数といいます。ジニ係数の定義は、次の通りです。$\overline{x}$は平均値です。
# 
# \begin{eqnarray}
# GI=\sum_{i}\sum_{j}\left| \frac{x_i-x_j}{2n^2 \overline{x}}\right|
# \end{eqnarray}
# 
# ジニ係数は、ローレンツ曲線と45度線で囲まれた部分の面積の2倍の値で、0から1の値を取ります。値が大きければ大きいほど、不平等の度合いが大きくなります。この式を利用して、男女の一期目の成績について、ジニ係数をそれぞれ求めてください。

# In[ ]:


# 解答(2)
# ジニ係数計算するための関数
def heikinsa(data):
    subt = []
    for i in range(0, len(data)-1):
        for j in range(i+1, len(data)):
            subt.append(np.abs(data[i] - data[j]))
    return float(sum(subt))*2 / (len(data) ** 2)
    
def gini(heikinsa, data):
    return heikinsa / (2 * np.mean(data))

print('男性の数学の成績に関するジニ係数:', gini(heikinsa(np.array(sorted_data_G1_M)), np.array(sorted_data_G1_M)))
print('女性の数学の成績に関するジニ係数:', gini(heikinsa(np.array(sorted_data_G1_F)), np.array(sorted_data_G1_F)))
