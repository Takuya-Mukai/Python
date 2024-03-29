import sys
import os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from tensorflow.keras.datasets import mnist  # tf.kerasを使う場合（通常）
from PIL import Image
import pickle
from common.functions import sigmoid, softmax

# from keras.datasets import mnist
# tf.kerasではなく、Kerasを使う必要がある場合はこちらを有効にする


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data(normalize=True, one_hot_label=True):
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    x = np.shape(x_train)
    t = np.shape(t_train)
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    if normalize:
        x_train = x_train / 255
        x_test = x_test / 255

    if one_hot_label:
        t = np.zeros((t_train.size, 10))
        for i in range(t_train.size):
            t[i][t_train[i]] = 1
        t_train = t
        t = np.zeros((t_test.size, 10))
        for i in range(t_test.size):
            t[i][t_test[i]] = 1
        t_test = t

    return x_train, t_train, x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


# accuracy_cnt = 0
# x_train, t_train, x_test, t_test = get_data()
# # print(x_test.shape)
# # print(t_test.shape)
# x = x_test.reshape(10000, 784)/255
# t = t_test/255
# print(t.shape)
# # print(t_test[0:10])
# network = init_network()
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     if p != t[i]:
#         accuracy_cnt += 1
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
