import abc
import numpy as np
from NN.CoreUtility import *


class Model(abc.ABC):
    @abc.abstractmethod
    def add_layer(self):
        pass

    @abc.abstractmethod
    def train(self, x, y, lr):
        pass


# back-propagation is a way of dynamic programming?
class LinearModel(Model):
    def __init__(self):
        self.layers = []
        self.loss = SquareLoss()
        self.x = None
        self.output = None
        self.lossTrace = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        self.x = x
        cur_x = x
        for layer in self.layers:
            cur_x = layer.calculate(cur_x)
        self.output = cur_x

    # 2d matrix can perfectly handle x batch
    def backward(self, y, lr):
        self.loss.calculate(self.output, y)
        pre = self.loss.get_derive()
        for layer in reversed(self.layers):
            layer.update(pre, lr)
            pre = layer.get_derive()

    def train(self, x, y, lr=1.0):
        self.forward(x)
        self.backward(y, lr)
        self.lossTrace.append(np.trace(self.loss.loss))

    def print_detail_info(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("the x is: ")
        print(self.x)
        print("the output is: ")
        print(self.output)
        print("the loss is: ")
        print(self.loss.loss)

    def print_info(self):
        print("Total loss is: ")
        print(np.trace(self.loss.loss))

