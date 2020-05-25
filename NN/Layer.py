import abc
import numpy as np
from NN.CoreUtility import *


class Layer(abc.ABC):
    @abc.abstractmethod
    def calculate(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def get_derive(self):
        pass


# Here I assume the each data input is nx1 dimension and the are m inputs for each batches
class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.w = np.zeros([output_size, input_size])
        self.b = np.zeros([output_size, 1])
        self.activate = lambda t: 1 / (1 + np.exp(-t))
        self.derivative = lambda t: self.activate(t) * (1 - self.activate(t))
        self.dI = None
        self.partial = None
        self.x = None
        self.batch_size = None

    def calculate(self, x):
        stretcher = np.ones((1, x.shape[1]))
        pre_activate = np.dot(self.w, x) + np.dot(self.b, stretcher)
        self.partial = self.derivative(pre_activate)
        self.x = x
        self.batch_size = inputs.shape[1]
        return self.activate(pre_activate)

    def get_derive(self):
        return self.dI

    def update(self, pre, lr):
        # dB is the same dimension as pre
        d_b = np.multiply(pre, self.partial)
        d_w = np.dot(d_b, self.x.T)
        self.dI = np.dot(self.w.T, np.multiply(pre, self.partial))
        self.b = self.b - np.mean(d_b, axis=1, keepdims=True) * lr
        self.w = self.w - d_w / self.batch_size * lr

    def print_info(self):
        print("weight is: ")
        print(self.w)
        print("bias is: ")
        print(self.b)
        print("input deriv is: ")
        print(self.dI)


class SimpleRNN(Layer):
    def __init__(self, input_size, output_size, hidden_size, window_size):
        self.w_oh = np.zeros([output_size, hidden_size])
        self.b_o = np.zeros([output_size, 1])
        self.w_hh = np.zeros([hidden_size, hidden_size])
        self.w_hx = np.zeros([hidden_size, input_size])
        self.b_h = np.zeros([hidden_size, 1])
        self.h = np.zeros([hidden_size, 1])
        self.pre_do = []
        self.pre_h = []
        self.pre_x = []
        self.active_grad = []
        self.window_size = window_size
        self.dx = None
        self.activate_func = Sigmoid()
        self.to_activate = None
        self.pre_h.append(self.h)

    def calculate(self, x):
        to_active = np.dot(self.w_hh, self.h) + np.dot(self.w_hx, x) + self.b_h
        self.h = self.activate_func.activate(to_active)
        grad = self.activate_func.gradient(to_active)
        self.active_grad.append(grad)
        o = np.dot(self.w_oh, self.h) + self.b_o
        self.pre_x.append(x)
        self.pre_h.append(self.h)
        return o

    def update(self, d_o, lr = 1):
        self.pre_do.append(d_o)
        if len(self.pre_x) == self.window_size:
            pre_dh = np.zeros(self.h.shape)
            dw_oh = np.zeros(self.w_oh.shape)
            dw_hh = np.zeros(self.w_hh.shape)
            dw_hx = np.zeros(self.w_hx.shape)
            dbh = np.zeros(self.h.shape)
            dbo = np.zeros(self.b_o.shape)
            for i in reversed(range(self.window_size)):
                dbo += self.pre_do[i]
                dw_oh += np.dot(self.pre_do[i], self.pre_h[i + 1].T)
                dh = np.dot(self.w_oh.T, self.pre_do[i])
                dh += np.dot(self.w_hh.T, np.multiply(pre_dh, self.active_grad[i]))
                dh_raw = np.multiply(dh, self.active_grad[i])
                dw_hh += np.dot(dh_raw, self.pre_h[i].T)
                dw_hx += np.dot(dh_raw, self.pre_x[i].T)
                dbh += dh_raw
                pre_dh = dh
            self.w_hh -= dw_hh * lr
            self.w_hx -= dw_hx * lr
            self.w_oh -= dw_oh * lr
            self.b_o -= dbo * lr
            self.b_h -= dbh * lr
            self.pre_x.clear()
            self.pre_h.clear()
            self.pre_do.clear()
            self.active_grad.clear()
            self.pre_h.append(self.h)


    def get_derive(self):
        return self.dx
