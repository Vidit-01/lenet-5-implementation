from Layers import (
    Conv2D,
    Conv2D_Mapped,
    AvgPool2D,
    DenseLayer,
    Tanh,
    Sequence,
    Flatten
)
from LossFunctions import SoftmaxCrossEntropy
from Optimizers import SGD


class Lenet5(Sequence):
    def __init__(self):
        layers = [
            Conv2D(in_channels=1, out_channels=6, kernel_size=5),
            Tanh(),

            AvgPool2D((2,2),2),
            Tanh(),

            Conv2D(in_channels=6 ,out_channels=16, kernel_size=5),
            Tanh(),

            AvgPool2D((2,2),2),
            Tanh(),

            Conv2D(in_channels=16, out_channels=120, kernel_size=5),
            Tanh(),

            Flatten(),
            
            DenseLayer(120, 84),
            Tanh(),

            DenseLayer(84, 10)
        ]
        super().__init__(layers=layers,loss_fn=SoftmaxCrossEntropy(),optimizer=SGD())

        
