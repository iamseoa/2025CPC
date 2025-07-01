# CustomVGGNet.py
import cupy as cp
from models.layers import Conv2D, ReLU, Flatten, Linear, MaxPool2D, Dropout, BatchNorm2D

class VGGBlock:
    def __init__(self, in_channels, out_channels, num_convs):
        self.layers = []
        for _ in range(num_convs):
            self.layers.append(Conv2D(in_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(BatchNorm2D(out_channels))
            self.layers.append(ReLU())
            in_channels = out_channels
        self.layers.append(MaxPool2D(kernel_size=2))
        self.params = []
        for layer in self.layers:
            for attr in ['weight', 'bias', 'gamma', 'beta', 'running_mean', 'running_var']:
                if hasattr(layer, attr):
                    self.params.append(getattr(layer, attr))
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class CustomVGGNet:
    def __init__(self, num_classes=100):
        self.block1 = VGGBlock(3, 32, num_convs=1)
        self.block2 = VGGBlock(32, 64, num_convs=1)
        self.block3 = VGGBlock(64, 128, num_convs=2)
        self.block4 = VGGBlock(128, 256, num_convs=2)

        self.flatten = Flatten()
        self.fc1 = Linear(256 * 2 * 2, 256)
        self.relu1 = ReLU()
        self.drop1 = Dropout(0.5)
        self.fc2 = Linear(256, num_classes)

        self.layers = (
            [*self.block1.layers, *self.block2.layers, *self.block3.layers, *self.block4.layers,
             self.flatten, self.fc1, self.relu1, self.drop1, self.fc2]
        )

        self.params = []
        for block in [self.block1, self.block2, self.block3, self.block4]:
            self.params.extend(block.params)
        for layer in [self.fc1, self.fc2]:
            for attr in ['weight', 'bias']:
                if hasattr(layer, attr):
                    self.params.append(getattr(layer, attr))

    def forward(self, x):
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.drop1.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, dout):
        dout = self.fc2.backward(dout)
        dout = self.drop1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.block4.backward(dout)
        dout = self.block3.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.block1.backward(dout)
        return dout
