import abc
import numpy as np


class Loss(abc.ABC):
    @abc.abstractmethod
    def get_loss(self, output, target):
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
        return self.loss

    def get_derive(self):
        return self.derivative

    def get_loss(self):
        return self.loss


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
