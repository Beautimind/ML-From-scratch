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
        pre = self.loss.getDerive()
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

