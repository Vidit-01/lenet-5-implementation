from sklearn.datasets import fetch_openml
import numpy as np
from model import Lenet5

def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0   # normalize
    y = mnist.target.astype(np.int64)
    X = X.reshape(-1, 1, 28, 28)
    X_padded = np.pad(X, ((0,0), (0,0), (2,2), (2,2)), mode='constant')
    X_train, X_test = X_padded[:10000], X_padded[68000:]
    y_train, y_test = y[:10000], y[68000:]

    return X_train, y_train, X_test, y_test



def evaluate(model, X, y, batch_size=128):
    correct = 0
    total = 0

    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model.forward(xb)
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == yb)
        total += len(yb)

    return correct / total

def get_batches(X, y, batch_size):
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


X_train, y_train, X_test, y_test = load_mnist()
model = Lenet5()

epochs = 40
batch_size = 128

for epoch in range(epochs):
    loss = 0
    for Xb, yb in get_batches(X_train, y_train, batch_size):
        logits = model.forward(Xb)
        loss = model.backward(yb)
        model.step()

    print(f"Epoch {epoch+1} | Loss: {loss:.4f}, | Acc: {evaluate(model,X_train,y_train)}")

acc = evaluate(model, X_test, y_test)
print("Test Accuracy:", acc)