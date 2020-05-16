class Loss(abc.ABC):
    @abc.abstractmethod
    def getLoss(self, output, target):
        pass

    @abc.abstractmethod
    def getDerive(self):
        pass


# one reason for declare variable in init is that it is easier to find what variable your class contains
class SquareLoss(Loss):
    def __init__(self):
        self.derivative = None
        self.loss = None

    def calculate(self, output, target):
        self.derivative = output - target
        self.loss = np.dot(self.derivative.T, self.derivative) * 0.5

    def getDerive(self):
        return self.derivative

    def getLoss(self):
        return self.loss