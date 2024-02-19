import sys
import os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from tensorflow.keras.datasets import mnist  # tf.kerasを使う場合（通常）
from PIL import Image
import pickle
import common.function as sigmoid, softmax

# from keras.datasets import mnist
# tf.kerasではなく、Kerasを使う必要がある場合はこちらを有効にする
(x_train, t_train), (x_test, t_test) = mnist.load_data()
print(x_train.shape)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


copy_x_train = x_train.copy().reshape(60000, 784)
img_show(x_train[0])


def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a1 = np.dot(z1, W2) + b2
    z2 = sigmoid(a1)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y



