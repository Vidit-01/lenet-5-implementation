import numpy as np

class Tanh:
    def __init__(self):
        pass

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, d_out):
        # derivative = 1 - tanh^2(x)
        return d_out * (1 - self.out ** 2)
