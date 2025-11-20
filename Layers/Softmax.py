import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        # stable softmax
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, d_out):
        # general softmax backward: batch-wise Jacobian
        B, C = d_out.shape
        dX = np.zeros_like(d_out)

        for b in range(B):
            s = self.out[b].reshape(-1, 1)          # (C,1)
            jac = np.diagflat(s) - (s @ s.T)        # (C,C)
            dX[b] = jac @ d_out[b]

        return dX
