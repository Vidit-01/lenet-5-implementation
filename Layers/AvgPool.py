import numpy as np

class AvgPool2D:
    def __init__(self, pool_size, stride):
        self.ph, self.pw = pool_size
        self.s = stride           
        self.cache = None

    def forward(self, X):
        N, C, H_in, W_in = X.shape
        H_out = (H_in - self.ph) // self.s + 1
        W_out = (W_in - self.pw) // self.s + 1
        out = np.zeros((N, C, H_out, W_out))
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * self.s
                h_end = h_start + self.ph
                w_start = w * self.s
                w_end = w_start + self.pw
                X_slice = X[:, :, h_start:h_end, w_start:w_end]
                out[:, :, h, w] = np.mean(X_slice, axis=(2, 3))
        self.cache = X
        return out

    def backward(self, dOut):
        X = self.cache
        N, C, H_in, W_in = X.shape
        _, _, H_out, W_out = dOut.shape
        dX = np.zeros_like(X)
        distribution_factor = 1.0 / (self.ph * self.pw)

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * self.s
                h_end = h_start + self.ph
                w_start = w * self.s
                w_end = w_start + self.pw
                dOut_slice = dOut[:, :, h, w, np.newaxis, np.newaxis]
                dX_slice = dOut_slice * distribution_factor
                dX[:, :, h_start:h_end, w_start:w_end] += dX_slice


        return dX