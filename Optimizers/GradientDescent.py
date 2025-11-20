class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, param, grad):
        param -= self.lr * grad
