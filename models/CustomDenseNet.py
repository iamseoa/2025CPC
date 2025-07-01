# CustomDenseNet.py
import cupy as cp
from models.layers import Conv2D, ReLU, BatchNorm2D, AvgPool2D, Flatten, Linear

class DenseLayer:
    def __init__(self, in_channels, growth_rate):
        self.bn = BatchNorm2D(in_channels)
        self.relu = ReLU()
        self.conv = Conv2D(in_channels, growth_rate, kernel_size=3, padding=1)
        self.params = []
        for attr in ['gamma', 'beta']:
            if hasattr(self.bn, attr):
                self.params.append(getattr(self.bn, attr))
        if hasattr(self.bn, 'running_mean'):
            self.params.append(self.bn.running_mean)
        if hasattr(self.bn, 'running_var'):
            self.params.append(self.bn.running_var)
        for attr in ['weight', 'bias']:
            if hasattr(self.conv, attr):
                self.params.append(getattr(self.conv, attr))

    def forward(self, x):
        out = self.bn.forward(x)
        out = self.relu.forward(out)
        out = self.conv.forward(out)
        return cp.concatenate([x, out], axis=-1)

    def backward(self, dout):
        in_channels = self.bn.gamma.shape[0]
        dx = dout[..., :in_channels]
        dout_layer = dout[..., in_channels:]
        dout_layer = self.conv.backward(dout_layer)
        dout_layer = self.relu.backward(dout_layer)
        dout_layer = self.bn.backward(dout_layer)
        dx += dout_layer
        return dx

class DenseBlock:
    def __init__(self, num_layers, in_channels, growth_rate):
        self.layers = []
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(in_channels, growth_rate)
            self.layers.append(layer)
            in_channels += growth_rate
        self.out_channels = in_channels
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class TransitionLayer:
    def __init__(self, in_channels, out_channels):
        self.bn = BatchNorm2D(in_channels)
        self.relu = ReLU()
        self.conv = Conv2D(in_channels, out_channels, kernel_size=1)
        self.pool = AvgPool2D(kernel_size=2)
        self.params = []
        for attr in ['gamma', 'beta']:
            if hasattr(self.bn, attr):
                self.params.append(getattr(self.bn, attr))
        if hasattr(self.bn, 'running_mean'):
            self.params.append(self.bn.running_mean)
        if hasattr(self.bn, 'running_var'):
            self.params.append(self.bn.running_var)
        for attr in ['weight', 'bias']:
            if hasattr(self.conv, attr):
                self.params.append(getattr(self.conv, attr))

    def forward(self, x):
        x = self.bn.forward(x)
        x = self.relu.forward(x)
        x = self.conv.forward(x)
        x = self.pool.forward(x)
        return x

    def backward(self, dout):
        dout = self.pool.backward(dout)
        dout = self.conv.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.bn.backward(dout)
        return dout

class CustomDenseNet:
    def __init__(self, num_classes=100, growth_rate=12):
        self.block1 = DenseBlock(num_layers=4, in_channels=3, growth_rate=growth_rate)
        self.trans1 = TransitionLayer(self.block1.out_channels, 64)
        self.block2 = DenseBlock(num_layers=4, in_channels=64, growth_rate=growth_rate)
        self.trans2 = TransitionLayer(self.block2.out_channels, 128)
        self.block3 = DenseBlock(num_layers=4, in_channels=128, growth_rate=growth_rate)
        self.flatten = Flatten()
        self.fc = Linear(self.block3.out_channels, num_classes)

        self.layers = (
            self.block1.layers +
            [self.trans1] +
            self.block2.layers +
            [self.trans2] +
            self.block3.layers +
            [self.flatten, self.fc]
        )

        self.params = []
        self.params.extend(self.block1.params)
        self.params.extend(self.trans1.params)
        self.params.extend(self.block2.params)
        self.params.extend(self.trans2.params)
        self.params.extend(self.block3.params)
        for attr in ['weight', 'bias']:
            if hasattr(self.fc, attr):
                self.params.append(getattr(self.fc, attr))

    def forward(self, x):
        x = self.block1.forward(x)
        x = self.trans1.forward(x)
        x = self.block2.forward(x)
        x = self.trans2.forward(x)
        x = self.block3.forward(x)
        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.block3.backward(dout)
        dout = self.trans2.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.trans1.backward(dout)
        dout = self.block1.backward(dout)
        return dout

