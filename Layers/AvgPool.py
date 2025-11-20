import numpy as np

class AvgPool2D:
    def __init__(self, in_channels, kernel_size):
        self.k = kernel_size
        self.in_channels = in_channels
        
        # One W and b per channel (scalar params)
        self.W = np.ones((in_channels, 1))
        self.b = np.zeros((in_channels, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        B, C, H, W = X.shape
        k = self.k

        out_h = H // k
        out_w = W // k
        
        out = np.zeros((B, C, out_h, out_w))

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        region = X[b, c, i*k:(i+1)*k, j*k:(j+1)*k]
                        avg = np.mean(region)
                        out[b, c, i, j] = avg * self.W[c] + self.b[c]

        return out

    def backward(self, d_out):
        X = self.X
        B, C, H, W = X.shape
        k = self.k

        out_h = H // k
        out_w = W // k
        
        dX = np.zeros_like(X)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        region = X[b, c, i*k:(i+1)*k, j*k:(j+1)*k]

                        # gradient wrt W, b
                        avg = np.mean(region)
                        self.dW[c] += d_out[b, c, i, j] * avg
                        self.db[c] += d_out[b, c, i, j]

                        # gradient wrt X
                        dX[b, c, i*k:(i+1)*k, j*k:(j+1)*k] += \
                            (d_out[b, c, i, j] * self.W[c]) / (k*k)

        return dX
