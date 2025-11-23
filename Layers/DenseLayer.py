import numpy as np

class DenseLayer:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(1 / in_features)
        self.params = {}
        self.params['W'] = np.random.uniform(-limit, limit, (in_features, out_features))
        self.params['b'] = np.zeros((1, out_features))

    def forward(self, X):
        self.X = X        
        return X @ self.params['W'] + self.params['b']

    def backward(self, d_out):
        self.grads = {}
        self.grads['W'] = self.X.T @ d_out
        self.grads['b'] = np.sum(d_out, axis=0, keepdims=True)
        return d_out @ self.params['W'].T
