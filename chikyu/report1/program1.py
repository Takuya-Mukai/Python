import numpy as np
import matplotlib.pyplot as plt

n = 360
theta = np.linspace(0, 2 * np.pi, n+1)
  # 0 から 2*π まで (n+1)点のデータ(n分割)の配列を作る

k = 1
z = np.real(np.exp(1.j * k * theta))
k = 2
z = z + np.real(np.exp(1.j * k * theta)) # 波数1と2の重ね合わせ

plt.plot(theta, np.real(z)) # zの実部をグラフ表示
plt.xlim(0.0, 2 * np.pi) # 横軸の表示範囲の指定
plt.ylim(-2.0, 2.0) # 縦軸の表示範囲の指定
plt.pause(10.0) # 10秒間だけグラフを表示

