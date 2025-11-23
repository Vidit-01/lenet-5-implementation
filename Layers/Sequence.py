class Sequence:
    def __init__(self, layers, loss_fn=None, optimizer=None):
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.out = out
        return out

    def backward(self, y):
        loss = self.loss_fn.forward(self.out, y)
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss

    def step(self):
        if self.optimizer is None:
            return

        for layer in self.layers:
            if hasattr(layer, "params"):
                for name in layer.params:
                    self.optimizer.update(
                        layer.params[name],
                        layer.grads[name]
                    )
