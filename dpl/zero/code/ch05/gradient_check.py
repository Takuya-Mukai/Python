import numpy as np
import sys, os
sys.path.append(os.pardir)
from two_layer_net import TwoLayerNet
from ch03.mnist import get_data

x_train, t_train, x_test, t_test = get_data()
network = TwoLayerNet(input_size=784, output_size=50, hidden_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

gradient_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in gradient_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - gradient_numerical[key]))
    print(key + ":" + str(diff))
