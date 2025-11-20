import numpy as np

class DenseLayer:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(1 / in_features)
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X          # (B, in_features)
        return X @ self.W + self.b

    def backward(self, d_out):
        # d_out: (B, out_features)
        self.dW = self.X.T @ d_out
        self.db = np.sum(d_out, axis=0, keepdims=True)
        return d_out @ self.W.T
