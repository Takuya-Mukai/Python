import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.pardir)
from ch03.mnist import get_data
from two_layer_net import TwoLayerNet

x_train, t_tr, x_test, t_te = get_data()
x_train, x_train = (
    x_train / 255,
    x_test / 255,
)
t_train = np.zeros(len(t_tr) * 10)
t_test = np.zeros(len(t_te) * 10)

t_train = t_train.reshape(len(t_tr), 10)
t_test = t_test.reshape(len(t_te), 10)

print("shape of t_test is", t_test.shape)
for i in range(len(t_te)):
    t_test[i, t_te[i]] = 1
for i in range(len(t_tr)):
    t_train[i, t_tr[i]] = 1


train_loss_list = []
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    print("batch_mask is", batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(i, loss)

t = np.arange(0, len(train_loss_list))
plt.plot(t, train_loss_list)
plt.savefig("train.png")
