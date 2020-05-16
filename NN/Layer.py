import abc
import numpy as np


# support multiple input and add stochastic gradient decent

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
        self.inputs = None
        self.dI = None
        self.partial = None

    def calculate(self, inputs):
        stretcher = np.ones((1, inputs.shape[1]))
        toActive = np.dot(self.w, inputs) + np.dot(self.b, stretcher)
        self.partial = self.derivative(toActive)
        self.inputs = inputs
        self.batch_size = inputs.shape[1]
        return self.activate(toActive)

    def get_derive(self):
        return self.dI

    def update(self, pre, lr):
        # dB is the same dimension as pre
        dB = np.multiply(pre, self.partial)
        dW = np.dot(dB, self.inputs.T)
        self.dI = np.dot(self.w.T, np.multiply(pre, self.partial))
        self.b = self.b - np.mean(dB, axis=1, keepdims=True) * lr
        self.w = self.w - dW / self.batch_size * lr

    def printDebugInfo(self):
        print("weight is: ")
        print(self.w)
        print("bias is: ")
        print(self.b)
        print("input deriv is: ")
        print(self.dI)


class Loss(abc.ABC):
    @abc.abstractmethod
    def getLoss(self, output, target):
        pass

    @abc.abstractmethod
    def get_derive(self):
        pass


# one reason for declare variable in init is that it is easier to find what variable your class contains
class SquareLoss(Loss):
    def __init__(self):
        self.derivative = None
        self.loss = None

    def calculate(self, output, target):
        self.derivative = output - target
        self.loss = np.dot(self.derivative.T, self.derivative) * 0.5

    def get_derive(self):
        return self.derivative

    def getLoss(self):
        return self.loss


class Model(abc.ABC):
    @abc.abstractmethod
    def addLayer(self):
        pass

    @abc.abstractmethod
    def train(self, input, target, lr):
        pass


# back-propagation is a way of dynamic programming?
class LinearModel(Model):
    def __init__(self):
        self.layers = []
        self.loss = SquareLoss()
        self.input = None
        self.output = None
        self.lossTrace = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        self.input = input
        cur_input = input
        for layer in self.layers:
            cur_input = layer.calculate(cur_input)
        self.output = cur_input

    # 2d matrix can perfectly handle input batch
    def backward(self, target, lr):
        self.loss.calculate(self.output, target)
        pre = self.loss.get_derive()
        for layer in reversed(self.layers):
            layer.update(pre, lr)
            pre = layer.dI

    def train(self, input, target, lr):
        self.forward(input)
        self.backward(target, lr)
        self.lossTrace.append(np.trace(self.loss.loss))

    def printDetailDebugInfo(self):
        print("Debug info for this iteration:")
        print("the input is: ")
        print(self.input)
        print("the output is: ")
        print(self.output)
        print("the loss is: ")
        print(self.loss.loss)
        for layer in self.layers:
            layer.printDebugInfo()

    def printDebugInfo(self):
        print("Total loss is: ")
        print(np.trace(self.loss.loss))


class ActivateFunc(abc.ABC):
    @abc.abstractmethod
    def activate(self, x):
        pass

    @abc.abstractmethod

    def gradient(self, x):
        pass


class Sigmoid(ActivateFunc):
    def __init__(self):
        self.activate_fml = lambda t: 1 / (1 + np.exp(-t))
        self.gradient_fml = lambda t: self.activate(t) * (1 - self.activate(t))

    def activate(self, x):
        return self.activate_fml(x)

    def gradient(self, x):
        return self.gradient_fml(x)



    # use truncated-BPTT algorithm for weights update, use window_size to set how much record to consider each time


class SimpleRNN(Layer):
    def __init__(self, input_size, output_size, hidden_size, window_size):
        self.w_oh = np.zeros([output_size, hidden_size]);
        self.w_hh = np.zeros([hidden_size, hidden_size])
        self.w_hx = np.zeros([hidden_size, input_size])
        self.h = np.zeros([hidden_size, 1])
        self.pre_gradient_o = []
        self.pre_h = []
        self.pre_x = []
        self.window_size = window_size
        self.dx = None
        self.activate_func = Sigmoid()
        self.to_activate = None

    def calculate(self, inputs):
        self.h = self.w_hh * self.h + self.w_hx * inputs
        self.to_activate = self.w_oh * self.h
        self.pre_x.append(inputs)
        return self.activate_func.activate(self.to_activate)

    def update(self, pre):
        self.pre_h.append(self.h)
        d_o = np.multiply(pre, self.activate_func.gradient(self.to_activate))
        self.pre_gradient_o.append(d_o)
        self.dx = np.dot(np.dot(d_o.T, self.w_oh), self.w_hx).T
        if len(self.pre_h) == self.window_size:
            prefix_mat = np.identity(self.h.shape[0])
            dw_oh = np.zeros(self.w_oh.shape)
            dw_hh = np.zeros(self.w_hh.shape)
            dw_hx = np.zeros(self.w_hx.shape)
            for i in range(len(self.pre_h)):
                dw_oh += np.dot(self.pre_gradient_o[i], self.pre_h[i].T)
                temp = np.dot(self.pre_gradient_o[i].T, self.w_oh)
                dw_hh += np.dot(np.dot(prefix_mat, self.pre_h[i]), temp)
                dw_hx += np.dot(np.dot(prefix_mat, self.pre_x[i]), temp)
                prefix_mat = np.dot(prefix_mat, self.w_hh)
            self.w_oh += dw_oh
            self.w_hx += dw_hx
            self.w_hh += dw_hh

    def get_derive(self):
        return self.dx
