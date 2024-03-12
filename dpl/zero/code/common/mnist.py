import numpy as np
from tensorflow.keras.datasets import mnist  # tf.kerasを使う場合（通常）

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
