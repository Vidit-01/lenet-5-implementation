import numpy as np


def im2col(X, k):
    B, C, H, W = X.shape
    out_h = H - k + 1
    out_w = W - k + 1

    i0 = np.repeat(np.arange(k), k)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(k), k * C)
    j1 = np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)

    c = np.repeat(np.arange(C), k*k).reshape(-1,1)

    cols = X[:, c, i, j]
    return cols

def col2im(cols, X_shape, k):
    B, C, H, W = X_shape
    out_h = H - k + 1
    out_w = W - k + 1

    X_grad = np.zeros((B, C, H, W))

    i0 = np.repeat(np.arange(k), k)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(k), k * C)
    j1 = np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    c = np.repeat(np.arange(C), k*k).reshape(-1,1)

    np.add.at(X_grad, (slice(None), c, i, j), cols.reshape(B, -1, out_h*out_w))

    return X_grad




class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size

        limit = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
        self.params = {
            "W": np.random.uniform(-limit, limit,
                (out_channels, in_channels, kernel_size, kernel_size)),
            "b": np.zeros(out_channels)
        }
        self.grads = {
            "W": np.zeros_like(self.params["W"]),
            "b": np.zeros_like(self.params["b"])
        }

    def forward(self, X):
        X = np.ascontiguousarray(X)
        self.X = X
        B, C, H, W = X.shape
        k = self.k
        self.out_h = H - k + 1
        self.out_w = W - k + 1

        self.X_col = im2col(X, k)               
        self.X_col = self.X_col.transpose(0, 2, 1)  
        W_col = self.params['W'].reshape(self.out_ch, -1)

        out = self.X_col @ W_col.T
        out = out.transpose(0,2,1).reshape(B, self.out_ch, self.out_h, self.out_w)


        return out

    def backward(self, dout):
        X = np.ascontiguousarray(self.X)
        B = dout.shape[0]
        k = self.k
        C_out = self.out_ch

        dout_flat = dout.reshape(B, C_out, -1).transpose(0,2,1)

        dW_col = dout_flat.transpose(0,2,1) @ self.X_col 
        self.grads["W"] = dW_col.sum(axis=0).reshape(self.params["W"].shape)

        self.grads["b"] = dout.sum(axis=(0,2,3))

        W_col = self.params["W"].reshape(C_out, -1)
        dX_col = dout_flat @ W_col 
        dX_col = dX_col.transpose(0,2,1)

        dX = col2im(dX_col, self.X.shape, k)
        return dX
