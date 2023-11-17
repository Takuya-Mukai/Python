#!/usr/bin/env python
# coding: utf-8

# # Chapter 7 練習と総合問題解答

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
sns.set()
%matplotlib inline

# 小数第３位まで表示
%precision 3

# #### <練習問題 7-1>
# 
# 3章で使った、数学の成績を示すデータである「student-mat.csv」を使って、学校を選んだ理由（`reason`）を円グラフ化して、それぞれの割合を出してください。

# In[ ]:


# 解答
student_data_math = pd.read_csv('student-mat.csv',sep=';')
student_data_math.groupby('reason').size().plot(kind='pie', autopct='%1.1f%%',startangle=90)
plt.ylabel('')
plt.axis('equal')

# #### <練習問題 7-2>
# 
# <練習問題 7-2>と同じデータで、`higher` (高い教育を受けたいかどうか。値は`yes`か`no`）を軸にして、それぞれの数学の最終成績`G3`の平均値を棒グラフで表示してください。ここから何か推測できることはありますか？

# In[ ]:


# 解答
student_data_math.groupby('higher')['G3'].mean().plot(kind='bar')
plt.xlabel('higher')
plt.ylabel('G3 grade avg')

# 高い教育を受けた人たちの方が成績は高めであることがわかる

# #### <練習問題 7-3>
# 
# <練習問題 7-2>と同じデータで、通学時間（`traveltime`）を軸にして、それぞれの数学の最終成績`G3`の平均値を横棒グラフで表示してください。何か推測できることはありますか？

# In[ ]:


# 解答
student_data_math.groupby(['traveltime'])['G3'].mean().plot(kind='barh')
plt.xlabel('G3 Grade avg')

# 通学時間が長いと成績が低くなる傾向にある

# ## 7.5 総合問題

# ### ■総合問題7-1 時系列データ分析
# 
# ここでは、本章で身に付けたpandasやscipyなどを使って、時系列データついて扱っていきましょう。
# 
# （1）（データの取得と確認）下記のサイトより、dow_jones_index.zipをダウンロードし、含まれている`dow_jones_index.data`を使って、データを読み込み、はじめの5行を表示してください。またデータのそれぞれのカラム情報等を見て、`null`などがあるか確認してください。　　
# 
# https://archive.ics.uci.edu/ml/machine-learning-databases/00312/dow_jones_index.zip　　

# In[ ]:


# 解答 (1)
# データの取得
import requests, zipfile
from io import StringIO
import io

# url 
zip_file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00312/dow_jones_index.zip'
r = requests.get(zip_file_url, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
# 展開
z.extractall()

# In[ ]:


# データの読み込み
dow_jones_index = pd.read_csv('dow_jones_index.data',sep=',')

# In[ ]:


# 先頭5行の確認
dow_jones_index.head()

# In[ ]:


# データのカラム情報
dow_jones_index.info()

# （2）（データの加工）カラムの`open`、`high`、`low`、`close`等のデータは数字の前に$マークが付いているため、これを取り除いてください。また、日時を`date`型で読み込んでいない場合は、date型に変換しましょう。

# In[ ]:


# 解答 (2)
# 型変更　日時型
dow_jones_index.date = pd.to_datetime(dow_jones_index.date)

# ＄マークを消す
delete_dolchar = lambda x: str(x).replace('$', '')

# 対象は、open,high,low.close,next_weeks_open,next_weeks_close
# 文字型を数値型を変換する処理
dow_jones_index.open = pd.to_numeric(dow_jones_index.open.map(delete_dolchar))
dow_jones_index.high = pd.to_numeric(dow_jones_index.high.map(delete_dolchar))
dow_jones_index.low = pd.to_numeric(dow_jones_index.low.map(delete_dolchar))
dow_jones_index.close = pd.to_numeric(dow_jones_index.close.map(delete_dolchar))
dow_jones_index.next_weeks_open = pd.to_numeric(dow_jones_index.next_weeks_open.map(delete_dolchar))
dow_jones_index.next_weeks_close = pd.to_numeric(dow_jones_index.next_weeks_close.map(delete_dolchar))

# In[ ]:


# 再確認
dow_jones_index.head()

# （3）カラムの`close`について、各`stock`ごとの要約統計量を算出してください。

# In[ ]:


# 解答(3)
# indexをセットする
dow_jones_index_stock_index = dow_jones_index.set_index(['date','stock'])

# データフレームワークの再構成
dow_jones_index_stock_index_unstack = dow_jones_index_stock_index.unstack()

# closeのみ対象
dow_close_data = dow_jones_index_stock_index_unstack['close']

#　要約統計量
dow_close_data.describe()

# （4）カラムの`close`について、各`stock`の相関を算出する相関行列を出してください。

# In[ ]:


# 解答(4)
# 相関行列
corr_data = dow_close_data.corr()
corr_data

# （4 続き）また、`seaborn`の`heatmap`を使って、相関行列のヒートマップを描いてみましょう（ヒント：`pandas`の`corr()`を使います）。

# In[ ]:


# 解答(4)
# ヒートマップ
sns.heatmap(corr_data)

# （5）(4)で算出した相関行列の中で一番相関係数が高い`stock`の組み合わせを抽出してください。

# In[ ]:


# 解答(5)
# 相関係数が最大となるペアの抽出（自分自身以外の29ペアの中で）

# initial value
max_corr = 0
stock_1 = ''
stock_2 = ''

for i in range(0,len(corr_data)):
    print(
        corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]].idxmax()[1],
        corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]].idxmax()[0],
        corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]][0]
    )
    if max_corr < corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]][0]:
        max_corr = corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]][0]
        stock_1 = corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]].idxmax()[1]
        stock_2 = corr_data[i:i+1].unstack().sort_values(ascending=False)[[1]].idxmax()[0]

# max_coorのペアを出力
print('[Max Corr]:',max_corr)
print('[stock_1]:',stock_1)
print('[stock_2]:',stock_2)

# （5 続き）さらに、その中でもっとも相関係数が高いペアを抜き出し、それぞれの時系列グラフを描いてください。

# In[ ]:


# 解答(5) グラフ化
# ペアトレーディングなどに使われる。
dow_close_data_subsets =dow_close_data[[stock_1,stock_2]]
dow_close_data_subsets.plot(subplots=True,grid=True)
plt.grid(True)

# （6） pandasの`rolling()`（窓関数）を使って、上記で使った各`stock`ごとに、`close`の過去5期（5週間）移動平均時系列データを計算してください。

# >**[ポイント]**
# >
# >(6)、(7)の補足についての補足です。
# >
# >時系列データ$(\cdots ,y_{t-1},y_t,y_{t+1}, \cdots )$の過去n期の移動平均データとは、過去5期のデータの平均、つまり以下を意味します。

# \begin{eqnarray}
# ma_t = \sum_{s=t-n+1}^t \frac{y_s}{n}
# \end{eqnarray}

# >時系列データ$(\cdots ,y_{t-1},y_t,y_{t+1}, \cdots )$の前期（1週前）との比の対数時系列データとは、$\log \frac{y_t} {y_{t-1}}$から成るデータのことです。増減率$r_t = \frac{y_t - y_{t-1}}{y_t}$が小さいとき、$r_t \approx \log \frac{y_t} {y_{t-1}}$の関係が成り立ちます。これは、$x$が十分小さいときに成り立つ、$\log (1+x) \approx x$から導かれます。増減率データ$(r_1,\cdots ,r_N )$のボラティリティとは、標準偏差

# \begin{eqnarray}
# \sqrt{\frac{1}{N}\sum_{t=1}^N (r_t - \frac{1}{N}\sum_{t=1}^N r_t)^2}
# \end{eqnarray}

# >のことで、価格変動の大きさを示す指標として利用されます。

# In[ ]:


# 解答(6)
# 窓関数
dow_close_data.rolling(center=False,window=5).mean().head(10)

# （7） pandasの`shift()`を使って、上記で使った各stockごとに、`close`の前期（1週前）との比の対数時系列データを計算してください。さらに、この中で、一番ボラティリティ（標準偏差）が一番大きい`stock`と小さい`stock`を抜き出し、その対数変化率グラフを書いてください。

# In[ ]:


# 解答(7)
# 前週比（１期ずらし）をしたい場合、shiftを使う
# loopなどを使うより、断然処理が速い
log_ratio_stock_close = np.log(dow_close_data/dow_close_data.shift(1))

max_vol_stock = log_ratio_stock_close.std().idxmax()
min_vol_stock = log_ratio_stock_close.std().idxmin()

# 最大と最小の標準偏差のstock
print('max volatility:',max_vol_stock)
print('min volatility:',min_vol_stock)

#　グラフ化
log_ratio_stock_close[max_vol_stock].plot()
log_ratio_stock_close[min_vol_stock].plot()
plt.ylabel('log ratio')
plt.legend()
plt.grid(True)

# ### ■総合問題7-2 マーケティング分析
# 
# 次は、マーケティング分析でよく扱われる購買データです。一般ユーザーとは異なる法人の購買データですが、分析する軸は基本的に同じです。
# 
# （1）下記のURLで公開されているデータをpandasで読み込んでください（件数50万以上のデータで比較的大きいため、少し時間がかかります）。
# 
# http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
#     

# >**[ヒント]**
# >
# >`pd.ExcelFile`を使って、シートを`.parse('Online Retail')`で指定してください。

# In[ ]:


# 解答 (1)
#　時間がかかります
file_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
online_retail_data = pd.ExcelFile(file_url)

# シートを指定する
online_retail_data_table = online_retail_data.parse('Online Retail')
online_retail_data_table.head()

# In[ ]:


# データの確認
online_retail_data_table.info()

# In[ ]:


# 解答 (1)
# InvoiceNoの1文字目を抽出する処理。mapとLambda関数を使う
online_retail_data_table['cancel_flg'] = online_retail_data_table.InvoiceNo.map(lambda x:str(x)[0])
online_retail_data_table.groupby('cancel_flg').size()

# また、今回の分析対象は、`CustomerID`にデータが入っているレコードのみ対象にするため、そのための処理をしてください。さらに、カラムの`InvoiceNo`には数字の前に`C`があるものはキャンセルのため、このデータを取り除いてください。他にもデータとして取り除く必要なものがあれば、適宜処理してください。以下、このデータをベースに分析していきます。

# In[ ]:


# 解答　(1)
# 「C」から始まるものはキャンセルデータなので、取り除く処理を書く
# 「A」も異常値として処理して、削除する
# 上記の結果から、今回は先頭が「5」であるものだけを分析対象とする
# さらに、CustomerIDがあるデータだけを対象とする
online_retail_data_table = online_retail_data_table[(online_retail_data_table.cancel_flg == '5') & (online_retail_data_table.CustomerID.notnull())]

# （2）このデータのカラムには、購買日時や商品名、数量、回数、購買者の`ID`などがあります。ここで、購買者（`CustomerID`）のユニーク数、バスケット数（`InvoiceNo`のユニーク数）、商品の種類（`StockCode`ベースと`Description`ベースのユニーク数）を求めてください。

# In[ ]:


# 解答　(2)
# unique ID
print('購買者数（ユニーク）:',len(online_retail_data_table.CustomerID.unique()))

# unique StockCode
print('商品コード数:',len(online_retail_data_table.StockCode.unique()))

# unique description
# 上より多いから、同じstockcodeで違う名前になった商品がある。
print('商品名の種類数:',len(online_retail_data_table.Description.unique()))

# unique bascket
print('バスケット数:',len(online_retail_data_table.InvoiceNo.unique()))

# （3）このデータのカラムには、`Country`があります。このカラムを軸に、それぞれの国の購買合計金額（単位あたりの金額×数量の合計）を求め、降順にならべて、上位5つの国の結果を表示してください。

# In[ ]:


# 解答 (3)
# 売り上げ合計を求めるため、新しいカラムの追加（売り上げ＝数量×単価）
online_retail_data_table['TotalPrice'] = online_retail_data_table.Quantity * online_retail_data_table.UnitPrice

#　それぞれの国ごとに売り上げ合計金額を算出
country_data_total_p = online_retail_data_table.groupby('Country')['TotalPrice'].sum()

# 値に対して、降順にソートして、TOP5を抜き出す。
top_five_country =country_data_total_p.sort_values(ascending=False)[0:5]

# TOP5の国
print(top_five_country)

# TOP5の国のリスト
print('TOP5の国のリスト:',top_five_country.index)

# （4）上の上位5つの国について、それぞれの国の商品売り上げ（合計金額）の月別の時系列推移をグラフにしてください。ここで、グラフは分けて表示してください。

# In[ ]:


# 解答 (4)
# TOP5だけのデータを作成。
top_five_country_data = online_retail_data_table[online_retail_data_table['Country'].isin(top_five_country.index)]

# date と国ごとの売り上げ
top_five_country_data_country_totalP =top_five_country_data.groupby(['InvoiceDate','Country'],as_index=False)['TotalPrice'].sum()

# In[ ]:


# 解答 (4)
# TOP 5の売り上げ月別推移

# indexの設定（日時と国）
top_five_country_data_country_totalP_index=top_five_country_data_country_totalP.set_index(['InvoiceDate','Country'])

# 再構成
top_five_country_data_country_totalP_index_uns = top_five_country_data_country_totalP_index.unstack()

# resampleで時系列のデータを月別や四半期等に変更できる。今回は、月別(M)の合計を算出。そのあと、グラフ化
top_five_country_data_country_totalP_index_uns.resample('M').sum().plot(subplots=True,figsize=(12,10))

# グラフが被らないように
plt.tight_layout()

# （5）上の上位5つの国について、それぞれの国における商品の売り上げトップ5の商品を抽出してください。また、それらを国ごとに円グラフにしてください。なお、商品は「`Description`」ベースで集計してください。

# In[ ]:


# 解答 (5)
for x in top_five_country.index:
    #print('Country:',x)
    country = online_retail_data_table[online_retail_data_table['Country'] == x]
    country_stock_data = country.groupby('Description')['TotalPrice'].sum()
    top_five_country_stock_data=pd.DataFrame(country_stock_data.sort_values(ascending=False)[0:5])    
    plt.figure()
    plt.pie(
        top_five_country_stock_data,
        labels=top_five_country_stock_data.index,
        counterclock=False,
        startangle=90,
        autopct='%.1f%%',
        pctdistance=0.7
    )
    plt.ylabel(x)
    plt.axis('equal')
    #print(top_five_country_stock_data)

# ※補足：他、余力がある方は以下の課題に取り組んでみてください。なお、大学の講座等の課題に使っているため解答は省略しますので、あらかじめご了承ください。
# 
# 追加課題：購買者（CustomerID）の各合計購買金額を算出し、さらに金額をベースに降順に並び替えをします。カラムがCustomerIDと合計金額のあるテーブルを作成してください。そこから、購買者を10等分にグループ分けします（例：100人いたら、10人ずつにグループ分けします。）。それぞれのグループでの合計購買金額の範囲と、それぞれの金額合計値を算出してください（このアプローチを**デシル分析**といいます。）。この結果を用いて、パレートの法則（上位2割の顧客が売上全体の8割を占める）を確かめるため、それぞれのグループが売上の何割を占めるか計算してください。なお、マーケティング戦略では、このように顧客を分けることを**セグメンテーション**といい、上位2割に絞ってアプローチを仕掛けることを**ターゲティング**といいます。それぞれの戦略によりますが、優良顧客に的を絞った方が投資対効果が高いことが多いため、このようなアプローチを取ることがあります。
# 
# ヒントは、6章で学んだビン分割などを使います。
