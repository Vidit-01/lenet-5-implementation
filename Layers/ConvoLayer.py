import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size

        # Xavier init
        limit = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.uniform(-limit, limit, 
                    (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X):
        # X: (B, C, H, W)
        self.X = X
        B, C, H, W = X.shape
        k = self.k

        out_h = H - k + 1
        out_w = W - k + 1

        out = np.zeros((B, self.out_ch, out_h, out_w))

        for b in range(B):
            for oc in range(self.out_ch):
                for ic in range(self.in_ch):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = X[b, ic, i:i+k, j:j+k]
                            out[b, oc, i, j] += np.sum(region * self.W[oc, ic])
                out[b, oc] += self.b[oc]

        return out

    def backward(self, d_out):
        X = self.X
        B, C, H, W = X.shape
        k = self.k

        out_h = H - k + 1
        out_w = W - k + 1

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dX = np.zeros_like(X)

        for b in range(B):
            for oc in range(self.out_ch):
                for ic in range(self.in_ch):

                    for i in range(out_h):
                        for j in range(out_w):
                            region = X[b, ic, i:i+k, j:j+k]

                            self.dW[oc, ic] += d_out[b, oc, i, j] * region
                            dX[b, ic, i:i+k, j:j+k] += d_out[b, oc, i, j] * self.W[oc, ic]

                self.db[oc] += np.sum(d_out[b, oc])

        return dX
