# CustomResNet.py
import cupy as cp
from models.layers import Conv2D, BatchNorm2D, ReLU, GlobalAvgPool2D, Flatten, Linear

class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.equal_in_out = (in_channels == out_channels) and (stride == 1)
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        if not self.equal_in_out:
            self.shortcut_conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2D(out_channels)

        self.layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2]
        if not self.equal_in_out:
            self.layers += [self.shortcut_conv, self.shortcut_bn]
        self.params = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                self.params.extend(layer.params)

    def forward(self, x):
        self.identity = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        if self.equal_in_out:
            shortcut = self.identity
        else:
            shortcut = self.shortcut_conv.forward(self.identity)
            shortcut = self.shortcut_bn.forward(shortcut)
        out += shortcut
        self.out = out
        out = self.relu1.forward(out)
        return out

    def backward(self, dout):
        dout = self.relu1.backward(dout)
        dshortcut = dout.copy()
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        if self.equal_in_out:
            dx = dout + dshortcut
        else:
            dshortcut = self.shortcut_bn.backward(dshortcut)
            dshortcut = self.shortcut_conv.backward(dshortcut)
            dx = dout + dshortcut
        return dx

class CustomResNet:
    def __init__(self, num_classes=100):
        self.conv1 = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(16)
        self.relu = ReLU()

        # Stage 1:16→16 (3 blocks)
        self.layer1 = [ResidualBlock(16, 16, stride=1) for _ in range(3)]
        self.layer2 = [ResidualBlock(16, 32, stride=2)]
        self.layer2 += [ResidualBlock(32, 32, stride=1) for _ in range(2)]
        self.layer3 = [ResidualBlock(32, 64, stride=2)]
        self.layer3 += [ResidualBlock(64, 64, stride=1) for _ in range(2)]

        self.gap = GlobalAvgPool2D()
        self.flatten = Flatten()
        self.fc = Linear(64, num_classes)

        self.layers = [self.conv1, self.bn1, self.relu]
        self.layers += self.layer1 + self.layer2 + self.layer3
        self.layers += [self.gap, self.flatten, self.fc]

        self.params = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                self.params.extend(layer.params)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        out = self.gap.forward(out)
        out = self.flatten.forward(out)
        out = self.fc.forward(out)
        return out

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.flatten.backward(dout)
        dout = self.gap.backward(dout)
        for block in reversed(self.layer3):
            dout = block.backward(dout)
        for block in reversed(self.layer2):
            dout = block.backward(dout)
        for block in reversed(self.layer1):
            dout = block.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout

