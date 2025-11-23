import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        pass

    def forward(self, logits, y_true):
        self.logits = logits
        B, C = logits.shape
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        log_likelihood = -np.log(self.probs[np.arange(B), y_true] + 1e-12)
        loss = np.mean(log_likelihood)
        self.y_true = y_true
        return loss

    def backward(self):
        B = self.logits.shape[0]
        grad = self.probs.copy()
        grad[np.arange(B), self.y_true] -= 1
        return grad / B
