# CustomMobileNet.py
import cupy as cp
from models.layers import col2im, im2col, Param, Conv2D, ReLU, Flatten, Linear, Dropout, BatchNorm2D, GlobalAvgPool2D

class DepthwiseConv2D:
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Param(cp.random.randn(in_channels, 1, kernel_size, kernel_size).astype(cp.float32) * (2. / (in_channels * kernel_size * kernel_size))**0.5)
        self.bias = Param(cp.zeros(in_channels, dtype=cp.float32))
        self.cache = None

    def forward(self, x):
        x_nchw = x.transpose(0, 3, 1, 2)
        N, C, H, W = x_nchw.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        col = im2col(x_nchw, k, s, p)
        out = cp.empty((col.shape[0], C), dtype=cp.float32)
        for c in range(C):
            out[:, c] = cp.dot(col[:, c * k * k:(c + 1) * k * k], self.weight.data[c, 0].reshape(-1)) + self.bias.data[c]
        out_h = (H + 2 * p - k) // s + 1
        out_w = (W + 2 * p - k) // s + 1
        out = out.reshape(N, out_h, out_w, C)
        self.cache = (x, col, out_h, out_w)
        return out

    def backward(self, dout):
        x, col, out_h, out_w = self.cache
        x_nchw = x.transpose(0, 3, 1, 2)
        N, C, H, W = x_nchw.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        dout_flat = dout.reshape(N * out_h * out_w, C)
        self.bias.grad = dout_flat.sum(axis=0)
        dW = cp.zeros_like(self.weight.data)
        for c in range(C):
            dW[c, 0] = cp.dot(col[:, c * k * k:(c + 1) * k * k].T, dout_flat[:, c]).reshape(k, k)
        self.weight.grad = dW
        dcol = cp.zeros_like(col)
        for c in range(C):
            dcol[:, c * k * k:(c + 1) * k * k] = dout_flat[:, c:c+1] @ self.weight.data[c, 0].reshape(1, -1)
        dx_nchw = col2im(dcol, (N, C, H, W), k, s, p)
        dx = dx_nchw.transpose(0, 2, 3, 1)
        return dx

class MobileNetV2Block:
    def __init__(self, in_channels, out_channels, stride, expansion=6):
        mid_channels = in_channels * expansion
        self.use_expand = expansion != 1
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand = None
        self.expand_bn = None
        self.expand_relu = None
        if self.use_expand:
            self.expand = Conv2D(in_channels, mid_channels, kernel_size=1)
            self.expand_bn = BatchNorm2D(mid_channels)
            self.expand_relu = ReLU()
        self.dwconv = DepthwiseConv2D(mid_channels, kernel_size=3, stride=stride, padding=1)
        self.dw_bn = BatchNorm2D(mid_channels)
        self.dw_relu = ReLU()
        self.project = Conv2D(mid_channels, out_channels, kernel_size=1)
        self.project_bn = BatchNorm2D(out_channels)
        self.params = []
        for layer in [self.expand, self.expand_bn, self.dwconv, self.dw_bn, self.project, self.project_bn]:
            if layer is not None:
                for attr in ['weight', 'bias', 'gamma', 'beta', 'running_mean', 'running_var']:
                    if hasattr(layer, attr):
                        self.params.append(getattr(layer, attr))

    def forward(self, x):
        out = x
        if self.use_expand:
            out = self.expand.forward(out)
            out = self.expand_bn.forward(out)
            out = self.expand_relu.forward(out)
        out = self.dwconv.forward(out)
        out = self.dw_bn.forward(out)
        out = self.dw_relu.forward(out)
        out = self.project.forward(out)
        out = self.project_bn.forward(out)
        if self.use_residual:
            out = out + x
        return out

    def backward(self, dout):
        dout_main = dout
        dout_res = dout if self.use_residual else 0
        dout = self.project_bn.backward(dout_main)
        dout = self.project.backward(dout)
        dout = self.dw_relu.backward(dout)
        dout = self.dw_bn.backward(dout)
        dout = self.dwconv.backward(dout)
        if self.use_expand:
            dout = self.expand_relu.backward(dout)
            dout = self.expand_bn.backward(dout)
            dout = self.expand.backward(dout)
        if self.use_residual:
            dout += dout_res
        return dout

class CustomMobileNet:
    def __init__(self, num_classes=100, dropout_p=0.3):
        self.stem = Conv2D(3, 32, kernel_size=3, stride=1, padding=1)
        self.stem_bn = BatchNorm2D(32)
        self.stem_relu = ReLU()
        self.blocks_cfg = [
            [32, 16, 1, 1],
            [16, 24, 2, 6],
            [24, 24, 1, 6],
            [24, 32, 2, 6],
            [32, 32, 1, 6],
            [32, 32, 1, 6],
            [32, 64, 2, 6],
            [64, 64, 1, 6],
            [64, 64, 1, 6],
            [64, 96, 1, 6],
            [96, 160, 2, 6],
            [160, 160, 1, 6],
            [160, 320, 1, 6],
        ]
        self.blocks = []
        for in_c, out_c, stride, exp in self.blocks_cfg:
            self.blocks.append(MobileNetV2Block(in_c, out_c, stride, exp))
        self.last_conv = Conv2D(320, 1280, kernel_size=1)
        self.last_bn = BatchNorm2D(1280)
        self.last_relu = ReLU()
        self.global_avg_pool = GlobalAvgPool2D()
        self.dropout = Dropout(dropout_p)
        self.fc = Linear(1280, num_classes)
        self.layers = [self.stem, self.stem_bn, self.stem_relu] + \
                      self.blocks + \
                      [self.last_conv, self.last_bn, self.last_relu, self.global_avg_pool, self.dropout, self.fc]
        self.params = []
        for l in self.layers:
            if hasattr(l, 'params'):
                self.params += l.params
            for attr in ['weight', 'bias', 'gamma', 'beta', 'running_mean', 'running_var']:
                if hasattr(l, attr):
                    self.params.append(getattr(l, attr))

    def forward(self, x):
        x = self.stem.forward(x)
        x = self.stem_bn.forward(x)
        x = self.stem_relu.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.last_conv.forward(x)
        x = self.last_bn.forward(x)
        x = self.last_relu.forward(x)
        x = self.global_avg_pool.forward(x)
        x = self.dropout.forward(x)
        x = self.fc.forward(x)
        return x

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = self.dropout.backward(dout)
        dout = self.global_avg_pool.backward(dout)
        dout = self.last_relu.backward(dout)
        dout = self.last_bn.backward(dout)
        dout = self.last_conv.backward(dout)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        dout = self.stem_relu.backward(dout)
        dout = self.stem_bn.backward(dout)
        dout = self.stem.backward(dout)
        return dout


