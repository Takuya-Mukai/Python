#!/usr/bin/env python
# coding: utf-8

# # Chapter 6 練習と総合問題解答

# In[ ]:


# 以下のライブラリを使うので、あらかじめ読み込んでおいてください
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series,DataFrame

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline

# 小数第３位まで表示
%precision 3

# #### <練習問題 6-1>
# 
# 次のデータに対して、`Kyoto`の列だけ抜き出してみましょう。

# In[ ]:


hier_data_frame1 = DataFrame(
    np.arange(12).reshape((3,4)),
    index = [['c','d','d'],[1,2,1]],
    columns = [
        ['Kyoto','Nagoya','Hokkaido','Kyoto'],
        ['Yellow','Yellow','Red','Blue']
    ]
)

hier_data_frame1.index.names = ['key1','key2']
hier_data_frame1.columns.names = ['city','color']
hier_data_frame1

# In[ ]:


# 解答
hier_data_frame1['Kyoto']

# #### <練習問題 6-2>
# 
# <練習問題 6-1>のデータに対して、`city`をまとめて列同士の平均値を出してください。

# In[ ]:


# 解答
# city列合計
hier_data_frame1.mean(level='city', axis=1)

# #### <練習問題 6-3>
# 
# <練習問題 6-1>のデータに対して、`key2`ごとに行の合計値を算出してみましょう。
# 

# In[ ]:


# 解答
# key2行合計
hier_data_frame1.sum(level='key2')

# #### <練習問題 6-4>
# 
# 下記の2つのデータテーブルに対して、`ID`の値が同じもの同士で内部結合してみましょう。

# In[ ]:


# データ4の準備
data4 = {
    'ID':['0','1','2','3','4','6','8','11','12','13'],
    'city':['Tokyo','Osaka','Kyoto','Hokkaido','Tokyo','Tokyo','Osaka','Kyoto','Hokkaido','Tokyo'],
    'birth_year':[1990,1989,1992,1997,1982,1991,1988,1990,1995,1981],
    'name':['Hiroshi','Akiko','Yuki','Satoru','Steeve','Mituru','Aoi','Tarou','Suguru','Mitsuo']
}
df4 = DataFrame(data4)
df4

# データ5の準備
data5 = {
    'ID':['0','1','3','6','8'],
    'math':[20,30,50,70,90],
    'English':[30,50,50,70,20],
    'sex':['M','F','F','M','M'],
    'index_num':[0,1,2,3,4]
}
df5 = DataFrame(data5)
df5

# In[ ]:


# 解答
pd.merge(df4, df5,on='ID')

# #### <練習問題 6-5>
# 
# <練習問題 6-4>のデータを使って、`attri_data_frame4`をベースに`attri_data_frame5`のテーブルを外部結合してみましょう。

# In[ ]:


# 解答
pd.merge(df4, df5, how='outer')

# #### <練習問題 6-6>
# <練習問題 6-4>のデータを使って、`attri_data_frame4`に対して、以下のデータを縦結合してみましょう。

# In[ ]:


# データの準備
data6 = {
    'ID':['70','80','90','120','150'],
    'city':['Chiba','Kanagawa','Tokyo','Fukuoka','Okinawa'],
    'birth_year':[1980,1999,1995,1994,1994],
    'name':['Suguru','Kouichi','Satochi','Yukie','Akari']
}
df6 = DataFrame(data6)

# In[ ]:


pd.concat([df4,df6])

# #### <練習問題 6-7>
# 
# 3章で使用した数学の成績を示すデータである「student-mat.csv」を読み込み、年齢（`age`）を2倍にしたカラムを末尾に追加してみましょう。

# In[ ]:


# 解答
# データがあるディレクトリに、カレントディレクトリを移動してください
import pandas as pd
student_data_math = pd.read_csv('student-mat.csv',sep=';')
student_data_math['age_d'] = student_data_math['age'].map(lambda x: x*2)
student_data_math.head()

# #### <練習問題 6-8>
# 
# <練習問題 6-7>と同じデータで、「`absences`」のカラムについて、以下の3つのビンに分けてそれぞれの人数を数えてみましょう。なお、`cut`のデフォルトの挙動は右側が閉区間です。今回は、`cut`に対視して`right=False`のオプションを指定して、右側を開区間としてください。

# In[ ]:


#　分割の粒度
absences_bins = [0,1,5,100]

# In[ ]:


# 解答
student_data_math_ab_cut_data = pd.cut(student_data_math.absences,absences_bins,right=False)
pd.value_counts(student_data_math_ab_cut_data)

# #### <練習問題 6-9>
# 
# <練習問題 6-7>と同じデータで、「`absences`」のカラムについて、`qcut`を用いて3つのビンに分けてみましょう。

# In[ ]:


# 解答
student_data_math_ab_qcut_data = pd.qcut(student_data_math.absences,3)
pd.value_counts(student_data_math_ab_qcut_data)

# #### <練習問題 6-10>
# 
# <練習問題 6-7>で使用した「student-mat.csv」を使って、pandasの集計処理をしてみましょう。まずは、学校（`school`）を軸にして、`G1`の平均点をそれぞれ求めてみましょう。

# In[ ]:


# 解答
student_data_math = pd.read_csv('student-mat.csv',sep=';')
student_data_math.groupby(['school'])['G1'].mean()

# #### <練習問題 6-11>
# 
# <練習問題 6-7>で使用した「student-mat.csv」を使って、学校（`school`）と性別（`sex`）を軸にして、`G1`、`G2`、`G3`の平均点をそれぞれ求めてみましょう。

# In[ ]:


# 解答
student_data_math.groupby(['school','sex'])['G1','G2','G3'].mean()

# なお、<練習問題 6-10>の計算結果と表示が異なるのは、<練習問題 6-10>の解答がSeries型で今回の解答がDataFrame型だからです。

# #### <練習問題 6-12>
# 
# <練習問題 6-7>で使用した「student-mat.csv」を使って、学校（`school`）と性別（`sex`）を軸にして、`G1`、`G2`、`G3`の最大値、最小値をまとめて算出してみましょう。

# In[ ]:


# 解答
functions = ['max','min']
student_data_math2 = student_data_math.groupby(['school','sex'])
student_data_math2['G1','G2','G3'].agg(functions)

# #### <練習問題 6-13>
# 
# 以下のデータに対して、1列でもNaNがある場合は削除し、その結果を表示してください。

# In[ ]:


# データの準備
from numpy import nan as NA

df2 = pd.DataFrame(np.random.rand(15,6))

# NAにする
df2.ix[2,0] = NA
df2.ix[5:8,2] = NA
df2.ix[7:9,3] = NA
df2.ix[10,5] = NA


df2

# In[ ]:


# 解答
df2.dropna()

# #### <練習問題 6-14>
# 
# <練習問題 6-13>で準備したデータに対して、`NaN`を0で埋めてください。

# In[ ]:


# 解答
df2.fillna(0)

# #### <練習問題 6-15>
# 
# <練習問題 6-13>で準備したデータに対して、NaNをそれぞれの列の平均値で埋めてください。

# In[ ]:


# 解答
df2.fillna(df2.mean())

# In[ ]:


# 確認用
df2.mean()

# #### <練習問題 6-16>
# 
# 下記のようにして読み込んだUSDJPYのデータであるfx_jpusdataを使って、年ごとの平均値の推移データを作成してください。

# In[ ]:


import pandas_datareader.data as pdr

start_date = '2001/1/2'
end_date = '2016/12/30'

fx_jpusdata = pdr.DataReader('DEXJPUS','fred',start_date,end_date)

# In[ ]:


# 解答
fx_jpusdata.resample('Y').mean().head()

# #### <練習問題 6-17>
# 
# <練習問題 6-16>で使用したfx_jpusdataを使って、20日間の移動平均データを作成してください。ただし`NaN`は削除してください。なお、レコードとして存在しないデータであれば、特に補填する必要はありません。

# In[ ]:


import pandas_datareader.data as pdr

start_date = '2001/1/2'
end_date = '2016/12/30'

fx_jpusdata = pdr.DataReader('DEXJPUS','fred',start_date,end_date)

# In[ ]:


# 解答
fx_jpusdata_rolling20 = fx_jpusdata.rolling(20).mean().dropna()
fx_jpusdata_rolling20.head()

# ## 6.5 総合問題

# ### ■総合問題6-1 データ操作
# 3章で使用した、数学の成績を示すデータである「student-mat.csv」を使って、以下の問いに答えてください。
# 
# (1) 上記のデータに対して、年齢（`age`）×性別（`sex`）で`G1`の平均点を算出し、縦軸が年齢（`age`）、横軸が性別（`sex`）となるような表（テーブル）を作成しましょう。
# 
# (2) (1)で表示した結果テーブルについて、NAになっている行（レコード）をすべて削除した結果を表示しましょう。

# In[ ]:


# 解答
# (1)
# 「student-mat.csv」を配置したディレクトリに、カレントディレクトリを移動して、以下を実行してください
student_data_math = pd.read_csv('student-mat.csv',sep=';')

student_data_math.groupby(['age','sex'])['G1'].mean().unstack()

# In[ ]:


# 解答
# (2)
student_data_math.groupby(['age','sex'])['G1'].mean().unstack().dropna()
