import numpy as np

class Conv2D_Mapped:
    def __init__(self, in_channels, out_channels, kernel_size, mapping):
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        self.mapping = mapping            # custom connection table

        # Create weight only for mapped connections
        self.W = {}
        for oc in range(out_channels):
            self.W[oc] = {}
            for ic in mapping[oc]:
                limit = np.sqrt(1 / (kernel_size * kernel_size))
                self.W[oc][ic] = np.random.uniform(
                    -limit, limit,
                    (kernel_size, kernel_size)
                )

        # Bias for each output channel
        self.b = np.zeros((out_channels, 1))

        # grads
        self.dW = {oc: {ic: np.zeros((kernel_size, kernel_size))
                        for ic in mapping[oc]}
                   for oc in range(out_channels)}
        self.db = np.zeros((out_channels, 1))

    def forward(self, X):
        self.X = X
        B, C, H, W = X.shape
        k = self.k

        out_h = H - k + 1
        out_w = W - k + 1

        out = np.zeros((B, self.out_ch, out_h, out_w))

        # Compute mapped convolution
        for b in range(B):
            for oc in range(self.out_ch):
                for ic in self.mapping[oc]:
                    for i in range(out_h):
                        for j in range(out_w):
                            region = X[b, ic, i:i+k, j:j+k]
                            out[b, oc, i, j] += np.sum(region * self.W[oc][ic])

                out[b, oc] += self.b[oc]

        return out

    def backward(self, d_out):
        X = self.X
        B, C, H, W = X.shape
        k = self.k

        out_h = H - k + 1
        out_w = W - k + 1

        dX = np.zeros_like(X)

        # Reset grads
        for oc in range(self.out_ch):
            for ic in self.mapping[oc]:
                self.dW[oc][ic].fill(0)
        self.db.fill(0)

        for b in range(B):
            for oc in range(self.out_ch):

                # db
                self.db[oc] += np.sum(d_out[b, oc])

                for ic in self.mapping[oc]:

                    for i in range(out_h):
                        for j in range(out_w):
                            region = X[b, ic, i:i+k, j:j+k]

                            # dW
                            self.dW[oc][ic] += d_out[b, oc, i, j] * region

                            # dX
                            dX[b, ic, i:i+k, j:j+k] += \
                                d_out[b, oc, i, j] * self.W[oc][ic]

        return dX
