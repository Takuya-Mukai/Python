#!/usr/bin/env python
# coding: utf-8

# # Chapter 1 練習と総合問題解答

# In[ ]:


# 小数第３位まで表示
%precision 3

# #### <練習問題 1-1>
# 
# ある文字列（Data Scienceなど）を変数として、それを1文字ずつ表示させるプログラムを書いてください。

# In[ ]:


# 解答
sampl_str = "Data Science"

for i in range(0,len(sampl_str)):
    print(sampl_str[i])

# In[ ]:


# 別解答
for i in sampl_str:
    print(i)

# In[ ]:


# 別解答(改行したくない場合)
for i in sampl_str:
    print(i,end = " ")

# #### <練習問題 1-2>
# 
# 1から50までの自然数の和を計算するプログラムを書いて、最後の計算結果を表示させるプログラムを書いてください。

# In[ ]:


# 解答

# 普通の方法
s = 0
for x in range(1,51):
    s += x
    # s = s + xでも可
print(s)

# In[ ]:


# sumを使う方法
print(sum(range(1,51)))

# In[ ]:


# forを使う方法 
print(sum(x for x in range(1,51)))

# ## 1.3 総合問題

# ### ■ 総合問題1-1 素数判定
# 
# （1）10までの素数を表示させるプログラムを書いてください。なお、素数とは、1とその数自身以外の約数をもたない正の整数のことをいいます。

# In[ ]:


n_list = range(2, 10 + 1)

for i in range(2, int(10 ** 0.5) + 1):
    # 2, 3, ... と順に割り切れるかを調べていく
    n_list = [x for x in n_list if (x == i or x % i != 0)]
        
for j in n_list:
    print(j)

# （2）（1）をさらに一般化して、`N`を自然数として、`N`までの素数を表示する関数を書いてください。

# In[ ]:


# 解答(1)と(2)

# 関数の定義
def calc_prime_num(N):
    n_list = range(2, N + 1)
    
    for i in range(2, int(N ** 0.5) + 1):
        # 2, 3, ... と順に割り切れるかを調べていく
        n_list = [x for x in n_list if (x == i or x % i != 0)]
            
    for j in n_list:
        print(j)

# 計算実行
calc_prime_num(10)
