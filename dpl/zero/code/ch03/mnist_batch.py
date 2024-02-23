import sys
import os
sys.path.append(os.pardir)

import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
improt pickle
from common.function import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    return x_test, t_test
