#!/usr/bin/env python
# coding: utf-8

# # Chapter 11 総合演習問題の解答例

# In[ ]:


# 以下は必要なライブラリのため、あらかじめ読み込んでおいてください。
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series,DataFrame
import pandas as pd
import time

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline

# 機械学習ライブラリ
import sklearn

# 小数第３位まで表示
%precision 3

# ## 11.1 総合演習問題（1）
# キーワード：教師あり学習、画像認識、複数カテゴリーの分類、混同行列

#  Scikit-learnの`sklearn.datasets`パッケージに入っている手書き数字のデータセットを下記のように読み込み、各数字（0〜9）を予測するモデルを構築しましょう。このデータは、手書きの数字で、0から9までの画像データです。以下の実装では、データを読み込み、サンプルとなる数字の画像データを表示しています。
#  
# 数字のラベル（目的変数）が`digits.target`で、そのデータの特徴量（説明変数）は`digits.data`です。ここで、このデータをテストデータと学習データに分けてモデルを構築し、混同行列の結果を表示させてください。その際、何度実行しても同じように分離されるように`train_test_split`のパラメータで`random_state=0`と設定してください。
# 
# また、いくつかモデルを作成し、比較してみてください。どのモデルを使いますか。

# In[ ]:


# 分析対象データ
from sklearn.datasets import load_digits

digits = load_digits()

# 画像の表示
plt.figure(figsize=(20,5))
for label, img in zip(digits.target[:10], digits.images[:10]):
    plt.subplot(1,10,label+1)
    plt.axis('off')
    plt.imshow(img,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Number:{0}'.format(label))

# ### **解答例**

# In[ ]:


# 解答
# データの分割（学習データとテストデータ分ける）
from sklearn.model_selection import train_test_split

# 混同行列
from sklearn.metrics import confusion_matrix

# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
# SVM
from sklearn.svm import LinearSVC
# 決定木
from sklearn.tree import  DecisionTreeClassifier
# k-NN
from sklearn.neighbors import  KNeighborsClassifier
# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 分析対象データ
from sklearn.datasets import load_digits
digits = load_digits()

# 説明変数
X = digits.data
# 目的変数
Y = digits.target

# 学習データとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0)

# 上記は、必要なモジュールやデータを読み込み、いつもと同じように、学習データとテストデータに分けています。
# 
# 以下では、その学習データとテストデータにおいて、それぞれの手書き数字でいくつあるかカウントしており、大きな偏りはないようです。

# In[ ]:


# データがアンバランスに分かれていないか確認
# train
print('train:',pd.DataFrame(y_train,columns=['label']).groupby('label')['label'].count())

# test
print('test:',pd.DataFrame(y_test,columns=['label']).groupby('label')['label'].count())

# それでは、それぞれの手法を用いて、モデル構築を実施し、それぞれの混同行列やスコアを見てみましょう。

# In[ ]:


# それぞれのモデルに対して繰り返し実行して確認する
for model in [LogisticRegression(),LinearSVC(), 
              DecisionTreeClassifier(),
              KNeighborsClassifier(n_neighbors=3),
              RandomForestClassifier()]:
    
    fit_model = model.fit(X_train,y_train)
    pred_y = fit_model.predict(X_test)   
    confusion_m = confusion_matrix(y_test,pred_y)
    print('confusion_matrix:')
    print(confusion_m)
    # __class__.__name__は、そのモデルのクラス名
    print('train:',fit_model.__class__.__name__ ,fit_model.score(X_train,y_train))
    print('test:',fit_model.__class__.__name__ , fit_model.score(X_test,y_test))
    print('===============================================================\n')

# 上記の結果より、テストデータにおけるスコは4つ目のK-NNが一番高くなりました。上記では、いろいろな手法について、特にパラメーターはデフォルトのままでしたが、余裕があればいろいろと調整してみましょう。

# ## 11.2 総合演習問題（2）
# キーワード：教師あり学習、回帰、複数モデルの比較

# 以下のデータを読み込み、アワビの年齢を予測するモデルを構築してみましょう。目的変数は「`Rings`」になります。 なお、英語ですが参考URL「B-26」に参考情報を挙げてあります。
# 
# http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data　

# ### **解答例**

# まずは、データを読み込み、どのようなデータがあるるか確認します。

# In[ ]:


# データの読み込み
abalone_data = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
    header=None,
    sep=',')

# 列にラベルの設定
abalone_data.columns=['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings']

# 先頭5行を表示
abalone_data.head()

# 以下は探索的にデータを見ています。まず、列同士の組み合わせの散布図を見てみます。対角線上にはヒストグラムが表示されます。
# 

# In[ ]:


sns.pairplot(abalone_data)

# 以下は箱ひげ図です。

# In[ ]:


# 箱ひげ図として表示する列を指定
abalone_data[['Length','Diameter','Height','Whole','Shucked','Viscera','Shell']].boxplot()

# グリッドを表示する
plt.grid(True)

# `Whole`の値に広がりがあるのがわかります。

# 基本統計量も確認しましょう。

# In[ ]:


abalone_data.describe()

# `Height`のデータに0もありますが、今回はそのままモデル構築を実施します。

# In[ ]:


# 解答

# 線形回帰モデル
from sklearn.linear_model import LinearRegression
# 決定木(回帰)
from sklearn.tree import  DecisionTreeRegressor
# k-NN
from sklearn.neighbors import  KNeighborsRegressor
# ランダムフォレスト
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

X = abalone_data.iloc[:,1:7]
Y = abalone_data['Rings']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0)

# 標準化のためのモジュール
from sklearn.preprocessing import StandardScaler

# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

for model in [LinearRegression(),
              DecisionTreeRegressor(),
              KNeighborsRegressor(n_neighbors=5),
              RandomForestRegressor()]:
    
    fit_model = model.fit(X_train_std,y_train)
    
    print('train:',fit_model.__class__.__name__ ,fit_model.score(X_train_std,y_train))
    print('test:',fit_model.__class__.__name__ , fit_model.score(X_test_std,y_test))

# 上記の学習データとテストデータのスコアを見比べてみればわかる通り、モデル（回帰木）によっては、過学習になっている（学習データでスコアが1、テストデータのスコアが0.02）のかよくわかります。
# 次は、参考ですが、k-NNのパラメータ`k`を変更させて検証してみましょう。8章の<練習問題 8-8>でも同じような実装をしましたので、そちらも参考にしてください。

# In[ ]:


# 解答
# k-NN
from sklearn.neighbors import  KNeighborsRegressor

from sklearn.model_selection import train_test_split

X = abalone_data.iloc[:,1:7]
Y = abalone_data['Rings']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0)

# 標準化のためのモジュール
from sklearn.preprocessing import StandardScaler

# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

training_accuracy = []
test_accuracy =[]

neighbors_settings = range(1,50)

for n_neighbors in neighbors_settings:
    clf = KNeighborsRegressor(n_neighbors=n_neighbors)
    clf.fit(X_train_std,y_train)
    
    training_accuracy.append(clf.score(X_train_std,y_train))
    
    test_accuracy.append(clf.score(X_test_std,y_test))
    
plt.plot(neighbors_settings, training_accuracy,label='training score')
plt.plot(neighbors_settings, test_accuracy,label='test score')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()

# `k`が増えるに従って改善しているようですが、`k=25`付近でスコア0.5ちょっとが限界のようです。

# ## 11.3 総合演習問題（3）
# キーワード：教師あり学習、分類、マーケティング分析、検証、混同行列、正解率、適合率、再現率、F1値、ROC曲線、AUC

# 9章で扱った、以下の金融機関のデータ（bank-full.csv）を読み込んで、後の問いに答えてください。
# 
# http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

# 1.  数値データ（`age,balance,day,duration,campaign,pdays,previous`）における基本統計量（レコード数、最大値、最小値、標準偏差など）を算出してください。  
# 2.  データの`'job','marital','education','default','housing','loan'`のそれぞれについて、預金を申し込む人、申し込まない人の人数を算出してください。　　
# 3.   `y`（預金を申し込む、申し込まない）を目的変数として、予測モデルを構築してください。モデルは複数（ロジスティック回帰、SVM、決定木、k-NN、ランダムフォレストなど）を試してください。ただし、テスト用にデータはあらかじめ抜いてください（その際、`train_test_split`のパラメータは`random_state=0`で設定してください）。     
# 4.   テスト用のデータを使って、それぞれのモデルの検証をしましょう。各モデルのテストデータにおける正解率、適合率、再現率、F1スコア、混同行列を表示してください。どのモデルを使いますか。  
# 5.   それぞれのモデルのROC曲線を描いて、AUCを算出し、比較できるようにしてください。

# ### **解答例**

# ＜例題１＞以下でWebからデータを取得しています。

# In[ ]:


import io
import zipfile
import requests

# データがあるurl の指定
zip_file_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
r = requests.get(zip_file_url, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# 次に、データを読み込み、どんなデータがあるか確認します。

# In[ ]:


banking_c_data = pd.read_csv('bank-full.csv',sep=';')
banking_c_data.head()

# 続いて数値データの統計量を算出します。

# In[ ]:


banking_c_data.describe()

# ＜例題２＞ `yes`と`no`でそれぞれの割合を算出してみます。

# In[ ]:


col_name_list = ['job','marital','education','default','housing','loan']
for col_name in col_name_list:
    print('---------------- ' + col_name + ' ----------------------')
    print(banking_c_data.groupby([col_name,'y'])['y'].count().unstack() / banking_c_data.groupby(['y'])['y'].count()*100)

# ＜例題３＞説明変数を選択し、ダミー変数`banking_c_data_dummy`として変換します。これについては、後述のコラムを参照してください。

# In[ ]:


banking_c_data_dummy = pd.get_dummies(banking_c_data[['job','marital','education','default','housing','loan']])
banking_c_data_dummy.head()

# 目的変数は「`y`」ですが、値として「`yes`」か「`no`」かの文字列をとります。これを数値として扱うため、`yes`のときは「1」、そうでないときは「0」のフラグ変数`flg`を作っておきます。

# In[ ]:


# 目的変数：flg立てをする
banking_c_data_dummy['flg'] = banking_c_data['y'].map(lambda x: 1 if x =='yes' else 0)

# 以下はモデリングをしています。ここでは説明変数として、「`age`」、「`balance`」、「`campaign`」を選択します。

# In[ ]:


# 解答
# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
# SVM
from sklearn.svm import LinearSVC
# 決定木
from sklearn.tree import  DecisionTreeClassifier
# k-NN
from sklearn.neighbors import  KNeighborsClassifier
# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# データの分割（学習データとテストデータ分ける）
from sklearn.model_selection import train_test_split

# 混同行列、その他の指標
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score

# 説明変数
X = pd.concat([banking_c_data_dummy.drop('flg', axis=1),banking_c_data[['age','balance','campaign']]],axis=1)
# 目的変数
Y = banking_c_data_dummy['flg']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, stratify = Y, random_state=0)

for model in [LogisticRegression(),LinearSVC(), 
              DecisionTreeClassifier(),
              KNeighborsClassifier(n_neighbors=5),
              RandomForestClassifier()]:
    
    fit_model = model.fit(X_train,y_train)
    pred_y = fit_model.predict(X_test)
    confusion_m = confusion_matrix(y_test,pred_y)
    
    print('train:',fit_model.__class__.__name__ ,fit_model.score(X_train,y_train))
    print('test:',fit_model.__class__.__name__ , fit_model.score(X_test,y_test))
    print('Confution matrix:\n{}'.format(confusion_m))
    print('適合率:%.3f' % precision_score(y_true=y_test,y_pred=pred_y))
    print('再現率:%.3f' % recall_score(y_true=y_test,y_pred=pred_y))
    print('F1値:%.3f' % f1_score(y_true=y_test,y_pred=pred_y))

# 上の結果から、決定木、k-NN、ランダムフォレストを選ぶことにします。

# ＜例題4＞決定木、k-NN、ランダムフォレストについてROC曲線とAUCを算出します。

# In[ ]:


from sklearn.metrics import roc_curve,roc_auc_score

for model in [DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=5)
             ,RandomForestClassifier()]:
    
    fit_model = model.fit(X_train,y_train)
    method = fit_model.__class__.__name__ 
    fpr,tpr,thresholds = roc_curve(y_test,fit_model.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test,fit_model.predict_proba(X_test)[:,1])

    plt.plot(fpr,tpr,label=method+', AUC:' + str(round(auc,3)))
    plt.legend(loc=4)

# モデルなし
plt.plot([0, 1], [0, 1],color='black', lw= 0.5, linestyle='--')

# グラフの右下にそれぞれのモデルのAUCが算出されており、ランダムフォレストが一番高いという結果になりました。

# ### コラム：ダミー変数と多重共線性（行列計算の数式等に慣れていない方はスキップしてください）
# 

# 上では、ダミー変数化したものをそのまますべて代入し、モデル構築をしました。しかし、これは果たしてよいのでしょうか。11.2「重回帰」で、多重共線性について触れましたが、ダミー変数を扱う場合の注意点を述べます。以下の例を考えて、数式的に見ていきましょう。

# `k`個の要素から構成されているカテゴリ変数をダミー変数にする際に、`k`個をそのままダミー変数に用いると多重共線性が発生することを以下の具体例を用いて示します。
# 
# あるスーパーマーケットの1日のアイスクリームの販売個数 $y$ をその日の平均気温 $x_1$、天気 $z$（晴れ、くもり、雨の3要素）を用いて重回帰分析で予測することを考えます。

# |データNo|$y$(個)|$x_1$(℃)|$z$|
# |:--:|:--:|:--:|:--:|
# |1|903|21|くもり|
# |2|1000|27|晴れ|
# |3|1112|22|雨|
# |4|936|19|くもり|
# |5|1021|23|晴れ|
# |$\vdots$|$\vdots$|$\vdots$|$\vdots$|
# |n|$y_n$|$x_n$|$z_n$|

# 天気$z$を$x_2$を晴れ、$x_3$をくもり、$x_4$を雨として、次のようにダミー変数化します。

# |データNo|$y$(個)|$x_1$(℃)|$z$|$x_2$|$x_3$|$x_4$|
# |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
# |1|903|21|くもり|0|1|0|
# |2|1000|27|晴れ|1|0|0|
# |3|1112|22|雨|0|0|1|
# |4|936|19|くもり|0|1|0|
# |5|1021|23|晴れ|1|0|0|
# |$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|$\vdots$|
# |n|$y_n$|$x_{1n}$|$z_n$|$x_{2n}$|$x_{3n}$|$x_{4n}$|

# このとき、ダミー変数の2つの値が分かれば、残ったダミー変数の値も分かるので、説明変数に3つすべてを含める必要はないと考えられます。
# 
# 実際、$x_4 = -x_2 - x_3 +1$という関係が成り立ちます。この関係から、$x_2$、$x_3$、$x_4$すべてを重回帰分析の説明変数に含めると最小二乗推定値が求まらないことを示します。
# 
# 重回帰式$y=b_0+b_1 x_1+b_2 x_2+b_3 x_3+b_4 x_4$を考えます。訓練データを用いて

# \begin{eqnarray}
# \boldsymbol{y}=\left( \begin{array}{c}
# y_{1} \\
# y_{2} \\
# \vdots \\
# y_{n}
# \end{array} \right),\,\,\,
# X=\left( \boldsymbol{1},\boldsymbol{x}_1,\boldsymbol{x}_2,\boldsymbol{x}_3,\boldsymbol{x}_4 \right)
# =\left( \begin{array}{ccccc}
# 1 & x_{11} & x_{21} & x_{31} & x_{41} \\
# \vdots & \vdots & \vdots & \vdots & \vdots \\
# 1 & x_{1n} & x_{2n} & x_{3n} & x_{4n}
# \end{array} \right)
# \end{eqnarray}

# とすると、係数$b_0,b_1,\cdots ,b_4$の最小二乗推定値は

# \begin{eqnarray}
# \left( \begin{array}{c}
# b_{0} \\
# b_{1} \\
# \vdots \\
# b_{4}
# \end{array} \right)
# =({}^t\!X X)^{-1} {}^t\!X \boldsymbol{y}
# \end{eqnarray}

# と表されます。しかし、今回 $x_4 = -x_2 - x_3 +1$という関係から、${}^t\!X X$の行列式が0となり逆行列が存在しないことが次のように示されます。

# \begin{eqnarray}
# |{}^t\!X X| = \left|
#     \begin{array}{cc}
#         \left( \begin{array}{c}
#           {}^t\!\boldsymbol{1} \\
#           {}^t\!\boldsymbol{x}_1 \\
#           {}^t\!\boldsymbol{x}_2 \\
#           {}^t\!\boldsymbol{x}_3 \\
#           {}^t\!\boldsymbol{x}_4 \\
#         \end{array} \right)
#         X
#     \end{array}
#   \right|
#   =
#   \left|
#     \begin{array}{c}
#         {}^t\!\boldsymbol{1} X \\
#         {}^t\!\boldsymbol{x}_1 X \\
#         {}^t\!\boldsymbol{x}_2 X \\
#         {}^t\!\boldsymbol{x}_3 X \\
#         {}^t\!\boldsymbol{x}_4 X \\
#     \end{array}
#   \right|
#   =
#   \left|
#     \begin{array}{c}
#         {}^t\!\boldsymbol{1} X \\
#         {}^t\!\boldsymbol{x}_1 X \\
#         {}^t\!\boldsymbol{x}_2 X \\
#         {}^t\!\boldsymbol{x}_3 X \\
#         {}^t\!\boldsymbol{x}_4 X +  {}^t\!\boldsymbol{x}_2 X + {}^t\!\boldsymbol{x}_3 X \\
#     \end{array}
#   \right|
#   =
#   \left|
#     \begin{array}{c}
#         {}^t\!\boldsymbol{1} X \\
#         {}^t\!\boldsymbol{x}_1 X \\
#         {}^t\!\boldsymbol{x}_2 X \\
#         {}^t\!\boldsymbol{x}_3 X \\
#         {}^t\!\boldsymbol{1} X \\
#     \end{array}
#   \right| 
#   =0
# \end{eqnarray}

# ここで、3つ目の等号では4行目に2行目、3行目を加えても行列式は変わらないという性質を用い、4つ目の等号では$x_4 = -x_2 - x_3 +1$という関係を用いました。このように、行列式が0となり、最小二乗推定値は存在しません。よって、最小二乗推定値を求めるためにはダミー変数を1つ抜く（たとえば$x_4$）必要があります。
# 
# 今回は要素が3つから構成されるカテゴリ変数を用いましたが、一般の`n`個の要素から構成されるカテゴリ変数でも`n`個すべてをダミー変数に使用してしまうと同様に行列式が0となることが示せます。
# 
# 重回帰分析等を実施するときには、多重共線性の問題がありますので、説明変数にカテゴリ変数を使う場合は注意しましょう。なお、pandasでダミー変数を作るための`get_dummies`関数には、`drop_first`という最初のダミー変数を取り除くパラメータがありますので、必要に応じて使ってください。
# 
# 上で説明した行列の参考としては、参考文献「A-38」「B-27」があります。
# 

# ## 11.4 総合演習問題（4）
# キーワード：教師あり学習、教師なし学習、ハイブリッドアプローチ

# 9章で扱ったload_breast_cancerを使って、さらに予測精度（正解率）を上げるモデルを作成してみましょう。15.3と同じく、テスト用にデータはあらかじめ抜いて検証してください。その際、`train_test_split`のパラメータを`random_state=0`に設定してください。      

# In[ ]:


# 前回の解答
# 標準化のためのモジュール
from sklearn.preprocessing import StandardScaler

# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=0)

# In[ ]:


# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# In[ ]:


from sklearn.metrics import confusion_matrix

model = LogisticRegression()
fit_model = model.fit(X_train_std,y_train)
print('train:',fit_model.__class__.__name__ ,fit_model.score(X_train_std,y_train))
print('test:',fit_model.__class__.__name__ , fit_model.score(X_test_std,y_test))

pred_y = fit_model.predict(X_test_std)
confusion_m = confusion_matrix(y_test,pred_y)

print('Confution matrix:\n{}'.format(confusion_m))

# データを標準化して、単純にモデルを当てはめるとテストデータで正解率95.8％でした。この結果を上回る方法を考えてみてください。

# ### **解答例**

# いろいろなアプローチがありますが、ここでは教師なし学習＋教師あり学習のハイブリッドアプローチでやってみましょう。まずはクラスタリングしてみます。

# クラスター数を5に設定して計算します。

# In[ ]:


# インポート
from sklearn.cluster import KMeans

# KMeansオブジェクトを初期化
kmeans_pp = KMeans(n_clusters=5)

# クラスターの重心を計算
kmeans_pp.fit(X_train_std)

# クラスター番号を予測
y_train_cl = kmeans_pp.fit_predict(X_train_std)

# 学習データでモデルを構築した結果を使って、テストデータに適応させます。

# In[ ]:


# テストデータでクラスター番号を予測
y_test_cl = kmeans_pp.fit_predict(X_test_std)

# モデリングで扱えるように、フラグを立てます。

# In[ ]:


# 学習データで、所属しているクラスターにフラグを立てる
cl_train_data = pd.DataFrame(y_train_cl,columns=['cl_nm']).astype(str)
cl_train_data_dummy = pd.get_dummies(cl_train_data)
cl_train_data_dummy.head()

# In[ ]:


# テストデータで、所属しているクラスターにフラグを立てる
cl_test_data = pd.DataFrame(y_test_cl,columns=['cl_nm']).astype(str)
cl_test_data_dummy = pd.get_dummies(cl_test_data)
cl_test_data_dummy.head()

# 次に、目的変数のデータと説明変数のデータをまとめます。

# In[ ]:


# 学習データでデータを結合
merge_train_data = pd.concat([
         pd.DataFrame(X_train_std),
         cl_train_data_dummy,
         pd.DataFrame(y_train,columns=['flg'])
    ], axis=1)

# テストデータでデータを結合
merge_test_data = pd.concat([
        pd.DataFrame(X_test_std),
        cl_test_data_dummy,
        pd.DataFrame(y_test,columns=['flg'])
    ], axis=1)

# In[ ]:


merge_train_data.head()

# 次に、主成分分析をかけて、どの要素数がスコアが良いか計算してみます。

# In[ ]:


from sklearn.metrics import confusion_matrix

model = LogisticRegression()
X_train_data = merge_train_data.drop('flg', axis=1)
X_test_data = merge_test_data.drop('flg', axis=1)

y_train_data = merge_train_data['flg']
y_test_data = merge_test_data['flg']

# 主成分分析
from sklearn.decomposition import PCA

best_score = 0
best_num = 0

for num_com in range(8):
    pca = PCA(n_components=num_com+1)
    pca.fit(X_train_data)
    X_train_pca = pca.transform(X_train_data)
    X_test_pca = pca.transform(X_test_data)

    logistic_model = model.fit(X_train_pca,y_train_data)
    
    train_score = logistic_model.score(X_train_pca,y_train_data)
    test_score = logistic_model.score(X_test_pca,y_test_data)
    
    
    if best_score < test_score:
        best_score = test_score
        best_num = num_com+1
        
print('best score:',best_score)
print('best num componets:',best_num)

# クラスター分析＋主成分分析の結果を利用して、正解率96.5％に改善しました。
# 
# 単に精度を上げるためだけではなく、マーケティング分析でも、教師なし学習＋教師あり学習のアプローチを使用することがあります。
# 
# 具体的には、教師なしのクラスター分析を使ってそれぞれのセグメントの特性を把握した後に、各セグメントに、どれくらいの割合で（ある商品を）購入する人、しない人がいるのか予測したい時に教師あり学習を使ったりします。これらのアプローチについては他にもいろいろとアイデアが考えられると思いますので、データ分析をする時に検討ください。

# ## 11.5 総合演習問題（5）
# キーワード：時系列データ、欠損データの補完、シフト、ヒストグラム、教師あり学習

# 以下のように、2001年1月2日から2016年12月30日までの為替データ（ドル/円レートのJPYUSDとユーロ/ドルレートのUSDEUR）を読み込み、問いに答えてください。なお、DEXJPUSとDEXUSEUがそれぞれJPYUSDとUSDEURに想定しています。

# 1. 読み込んだデータには、祝日や休日等による欠損（NaN）があります。その補完処理をするために、直近の前の日におけるデータで補完してください。ただし年月のデータがない場合もありますので、その場合、今回は無視してください（改めて日付データを作成して、分析をすることも可能ですが、今回はこのアプローチはとりません。）。  
# 2. 上記のデータで、各統計量の確認と時系列のグラフ化をしてください。  
# 3. 当日と前日における差分をとり、それぞれの変化率（当日-前日）/前日のデータをヒストグラムで表示してください。　　  
# 4. 将来の価格（例：次の日）を予測するモデルを構築してみましょう。具体的には、2016年11月を訓練データとして、当日の価格を目的変数として、前日、前々日、3日前の価格データを使ってモデル（線形回帰）を構築し、2016年12月をテストデータとして、検証してください。また、他の月や年で実施すると、どんな結果になりますか。   

# まず、以下を実行して、データをダウンロードしてください。

# In[ ]:


!pip install pandas-datareader

# 以下で、対象となる期間の為替データを読み込みます。

# In[ ]:


import pandas_datareader.data as pdr

start_date = '2001-01-02'
end_date = '2016-12-30'

fx_jpusdata = pdr.DataReader('DEXJPUS','fred',start_date,end_date)
fx_useudata = pdr.DataReader('DEXUSEU','fred',start_date,end_date)

# ### **解答例**

# ＜例題1＞読み込んだデータには`na`がありますので、`fillna`を使って（`ffill`をパラメータとして設定）、前の値で埋めることにします。

# In[ ]:


fx_jpusdata_full = fx_jpusdata.fillna(method='ffill')
fx_useudata_full = fx_useudata.fillna(method='ffill')

# ＜例題2＞それぞれの基本統計量を確認します。

# In[ ]:


print(fx_jpusdata_full.describe())
print(fx_useudata_full.describe())

# 時系列のデータですので、グラフにしてみましょう。

# In[ ]:


fx_jpusdata_full.plot()
fx_useudata_full.plot()

# それぞれのグラフに特徴があるようです。
# 
# ＜例題3＞次に、前日の値との比をとって、それぞれヒストグラムにしてみましょう。

# In[ ]:


fx_jpusdata_full_r = (fx_jpusdata_full - fx_jpusdata_full.shift(1)) / fx_jpusdata_full.shift(1)
fx_useudata_full_r = (fx_useudata_full - fx_useudata_full.shift(1)) / fx_useudata_full.shift(1)

# In[ ]:


fx_jpusdata_full_r.hist(bins=30)
fx_useudata_full_r.hist(bins=30)

# ＜例題4＞次に、前日だけではなく、2日前、3日前の値とも比べるため、そのデータセットを作成しましょう。

# In[ ]:


merge_data_jpusdata = pd.concat([
        fx_jpusdata_full,
        fx_jpusdata_full.shift(1),
        fx_jpusdata_full.shift(2),
        fx_jpusdata_full.shift(3)
    ], axis=1)
merge_data_jpusdata.columns =['today','pre_1','pre_2','pre_3']
merge_data_jpusdata_nona = merge_data_jpusdata.dropna()
merge_data_jpusdata_nona.head()

# それでは、早速、モデル構築をしてみましょう。

# In[ ]:


from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# モデル
from sklearn import linear_model

# モデルの初期化
l_model = linear_model.LinearRegression()

pre_term = '2016-11'
pos_term = '2016-12'
        
for pre_list in (['pre_1'],['pre_1','pre_2'],['pre_1','pre_2','pre_3']):

    print(pre_list)
    train = merge_data_jpusdata_nona[pre_term]
    X_train = pd.DataFrame(train[pre_list])
    y_train = train['today']

    test = merge_data_jpusdata_nona[pos_term]
    X_test = pd.DataFrame(test[pre_list])
    y_test = test['today']
    
    # モデルのあてはめ
    fit_model = l_model.fit(X_train,y_train)
    print('train:',fit_model.__class__.__name__ ,fit_model.score(X_train,y_train))
    print('test:',fit_model.__class__.__name__ , fit_model.score(X_test,y_test))

# 上記の結果より、訓練データとテストデータに大きな乖離があり、過学習になっているようです。他にも適合率や再現率等も見てください。為替のデータや金融商品の価格予測は困難だといわれており、機械学習以外にもさまざまなアプローチや研究がされています。

# ## 11.6 総合演習問題（6）
# キーワード：時系列データ、回帰分析

# 以下の米国の旅客飛行機のフライトデータ」を取得し、読み込んで以下の問いに答えてください。ただし、今回は2000年より前のデータ（1987～1999）を分析対象とします。
# 
# http://stat-computing.org/dataexpo/2009/the-data.html
# 

# 
# 1. データを読み込んだ後は、年（Year）×月（Month）の平均の出発遅延時間（DepDelay）を算出してください。何かわかることはありますか。  
# 2. 1で算出したデータについて、1月から12月までの結果を時系列の折れ線グラフにしてください。その時、年ごとに比較できるように、1つのグラフにまとめてください。1987年から1999年までのデータについて、それぞれの時系列グラフが並ぶイメージです。
# 
# 3. 各航空会社（UniqueCarrier）ごとの平均遅延時間を算出してください。また、出発地（Origin）、目的地（Dest）を軸にして、平均遅延時間を算出してください。  
# 4. 遅延時間を予測するための予測モデルを構築します。目的変数をDepDelay、説明変数をArrDelay（実際の到着時間？）とDistance（飛行距離？）にして、モデルを構築しましょう。

# ※ データが上記サイトからダウンロードできなくなりました。以下のサイトから、ダウンロードしてください。
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7

# ### **解答**

# ＜例題1＞以下の実装は、データが取得できたとして、該当のパスから、規則性のあるファイル名を探し出し、それらのデータをマージしています。なお、glob関数は、Unixシェルの規則を使ってファイル名等のパターンマッチをします。ただし、すべてのデータを処理するためには、ある程度のPCのスペックや環境によりますので、1980年代までのデータを対象にします。

# In[ ]:


# pathを入力
path =r'/Users/data' 

# データをマージするための処理
import glob
import pandas as pd


allFiles = glob.glob(path + '/198*.csv')
data_frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    print(file_)
    df = pd.read_csv(file_,index_col=None, header=0,encoding ='ISO-8859-1' )
    list_.append(df)
frame_198 = pd.concat(list_)

# 1000万行以上で、2-3Gほどあります。読み込んだらデータをチェックしましょう。

# In[ ]:


frame_198.head()

# In[ ]:


frame_198.info()

# 次に、月別のレコード数を見てみましょう。

# In[ ]:


frame_198.groupby('Month')['Month'].count()

# 遅延はDepDelayです。平均を月ごとに見ると以下になります。

# In[ ]:


frame_198.groupby('Month')['DepDelay'].mean()

# ＜例題2＞遅延を年別、月別推移をグラフで見てみましょう。どうなるでしょうか？

# In[ ]:


year_month_avg_arrdelay = frame_198.groupby(['Year','Month'])['ArrDelay'].mean()

# In[ ]:


pd.DataFrame(year_month_avg_arrdelay).unstack().T.plot(figsize=(10,6))
plt.legend(loc='best')
plt.grid(True)

# 毎年12月や1月にピークが来ています。年末年始に遅れが生じるのは、感覚的にも理解できます。また6月にもピークが来ています。遅延時間は季節性があるようです。

# ここでは実施しませんが、異常値等のチェックもしましょう。最大値が極端に大きかったり、最小値が極端に小さい時もあるようです。

# ＜例題3＞ 次に航空会社によって（UniqueCarrier）、ArrDelay（遅延）に違いはあるのでしょうか。

# In[ ]:


frame_198.groupby(['UniqueCarrier'])['ArrDelay'].mean()

# PI航空会社の遅延が目立っています。

# 次は、出発地や目的地による違いです。かなりばらつきがあるようです。

# In[ ]:


origin_avg_arrdelay = pd.DataFrame(frame_19.groupby(['Origin'])['ArrDelay'].mean()).reset_index()
origin_avg_arrdelay.head()

# In[ ]:


dest_avg_arrdelay = pd.DataFrame(frame_19.groupby(['Dest'])['ArrDelay'].mean()).reset_index()
dest_avg_arrdelay.head()

# ＜例題4＞ 次は、遅延時間を予測するための簡単なモデルを作成します。

# In[ ]:


analysis_data = frame_19[['DepDelay','Distance','ArrDelay']]

# 今回、NAは分析対象から外します。6章でも述べましたが、実務では、このような欠損データ等はどのように扱うかはきちんと確認、議論した上で進めてください。

# In[ ]:


analysis_data_full = analysis_data.dropna()

# In[ ]:


X = analysis_data_full[['DepDelay','Distance']]
Y = analysis_data_full['ArrDelay']

# In[ ]:


# データの分割（学習データとテストデータ分ける）
from sklearn.model_selection import train_test_split

# モデル
from sklearn import linear_model

# モデルのインスタンス
l_model = linear_model.LinearRegression()

# 学習データとテストデータ分ける
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5,random_state=0)

# モデルのあてはめ
fit_model = l_model.fit(X_train,y_train)
print('train:',fit_model.__class__.__name__ ,fit_model.score(X_train,y_train))
print('test:',fit_model.__class__.__name__ , fit_model.score(X_test,y_test))
 
# 偏回帰係数
print(pd.DataFrame({'Name':X.columns,
                    'Coefficients':fit_model.coef_}).sort_values(by='Coefficients') )

# 切片 
print(fit_model.intercept_)

# 他、Spark（Pyspark）でも計算できますので、余裕があればやってみてください。
