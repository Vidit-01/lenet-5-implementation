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


mapping_c3 = [
    [0,1,2],        # feature map 0
    [1,2,3],        # feature map 1
    [2,3,4],        # feature map 2
    [3,4,5],        # feature map 3
    [4,5,0],        # feature map 4
    [5,0,1],        # feature map 5

    [0,1,2,3],      # feature map 6
    [1,2,3,4],      # feature map 7
    [2,3,4,5],      # feature map 8
    [3,4,5,0],      # feature map 9
    [4,5,0,1],      # feature map 10
    [5,0,1,2],      # feature map 11

    [0,1,2,3,4],    # feature map 12
    [1,2,3,4,5],    # feature map 13
    [2,3,4,5,0],    # feature map 14
    [3,4,5,0,1],    # feature map 15
]

class Lenet5(Sequence):
    def __init__(self):
        layers = [
            Conv2D(in_channels=1, out_channels=6, kernel_size=5),
            Tanh(),

            AvgPool2D(kernel_size=2,in_channels=6),
            Tanh(),

            Conv2D_Mapped(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                mapping=mapping_c3
            ),
            Tanh(),

            AvgPool2D(kernel_size=2,in_channels=16),
            Tanh(),

            Conv2D(in_channels=16, out_channels=120, kernel_size=5),
            Tanh(),
            Flatten(),
            DenseLayer(120, 84),
            Tanh(),

            DenseLayer(84, 10)
        ]
        super().__init__(layers=layers,loss_fn=SoftmaxCrossEntropy,optimizer=SGD)

        
