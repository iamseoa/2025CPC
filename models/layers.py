import numpy as np
import cupy as cp

def im2col(x, k, stride, padding):
    N, C, H, W = x.shape
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1
    x_padded = cp.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    col = cp.zeros((N, C, k, k, out_h, out_w), dtype=x.dtype)
    for y in range(k):
        y_max = y + stride * out_h
        for x_ in range(k):
            x_max = x_ + stride * out_w
            col[:, :, y, x_, :, :] = x_padded[:, :, y:y_max:stride, x_:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, x_shape, k, stride, padding):
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1
    col = col.reshape(N, out_h, out_w, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    img = cp.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1), dtype=col.dtype)
    for y in range(k):
        y_max = y + stride * out_h
        for x_ in range(k):
            x_max = x_ + stride * out_w
            img[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]
    return img[:, :, padding:H + padding, padding:W + padding]

class Param:
    def __init__(self, data):
        self.data = data
        self.grad = None
    @property
    def size(self):
        return self.data.size
    @property
    def nbytes(self):
        return self.data.nbytes

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        std = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        w_data = cp.asarray(np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std)
        self.weight = Param(w_data)
        if bias:
            b_data = cp.asarray(np.zeros(out_channels, dtype=np.float32))
            self.bias = Param(b_data)
        else:
            self.bias = None
        # 파라미터 리스트 등록
        self.params = [self.weight]
        if self.bias is not None:
            self.params.append(self.bias)

    def forward(self, x):
        self.x_shape = x.shape
        x = x.transpose(0, 3, 1, 2)  # NHWC → NCHW
        self.x = x
        self.col = im2col(x, self.kernel_size, self.stride, self.padding)
        w_col = self.weight.data.reshape(self.out_channels, -1)
        out = cp.dot(self.col, w_col.T)
        if self.use_bias:
            out += self.bias.data[None, :]
        N, C, H, W = self.x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out.reshape(N, out_h, out_w, self.out_channels)

    def backward(self, dout):
        N, out_h, out_w, _ = dout.shape
        dout_reshaped = dout.reshape(-1, self.out_channels)
        dw = cp.dot(dout_reshaped.T, self.col).reshape(self.weight.data.shape)
        dx_col = cp.dot(dout_reshaped, self.weight.data.reshape(self.out_channels, -1))
        dx = col2im(dx_col, self.x.shape, self.kernel_size, self.stride, self.padding)
        if self.use_bias:
            db = cp.sum(dout_reshaped, axis=0)
            self.bias.grad = db
        self.weight.grad = dw
        return dx.transpose(0, 2, 3, 1)

class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = Param(cp.ones(num_features, dtype=cp.float32))
        self.beta = Param(cp.zeros(num_features, dtype=cp.float32))
        self.running_mean = Param(cp.zeros(num_features, dtype=cp.float32))
        self.running_var = Param(cp.ones(num_features, dtype=cp.float32))
        self.eps = eps
        self.momentum = momentum
        # 파라미터 리스트 등록
        self.params = [self.gamma, self.beta, self.running_mean, self.running_var]

    def forward(self, x, training=True):
        self.x = x
        if training:
            mean = cp.mean(x, axis=(0,1,2), keepdims=True)
            var = cp.var(x, axis=(0,1,2), keepdims=True)
            # running 통계는 .data로 접근
            self.running_mean.data[...] = self.momentum * mean + (1 - self.momentum) * self.running_mean.data
            self.running_var.data[...] = self.momentum * var + (1 - self.momentum) * self.running_var.data
            self.mean = mean
            self.var = var
        else:
            self.mean = self.running_mean.data
            self.var = self.running_var.data
        self.x_norm = (x - self.mean) / cp.sqrt(self.var + self.eps)
        return self.gamma.data * self.x_norm + self.beta.data

    def backward(self, dout):
        N, H, W, C = dout.shape
        x_mu = self.x - self.mean
        std_inv = 1. / cp.sqrt(self.var + self.eps)
        dx_norm = dout * self.gamma.data
        dvar = cp.sum(dx_norm * x_mu * -0.5 * std_inv**3, axis=(0,1,2), keepdims=True)
        dmean = cp.sum(dx_norm * -std_inv, axis=(0,1,2), keepdims=True) + dvar * cp.mean(-2. * x_mu, axis=(0,1,2), keepdims=True)
        dx = dx_norm * std_inv + dvar * 2 * x_mu / (N*H*W) + dmean / (N*H*W)
        self.gamma.grad = cp.sum(dout * self.x_norm, axis=(0,1,2))
        self.beta.grad = cp.sum(dout, axis=(0,1,2))
        return dx

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask

class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.orig_shape)

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        std = np.sqrt(2. / in_features)
        w_data = cp.asarray(np.random.randn(in_features, out_features).astype(np.float32) * std)
        self.weight = Param(w_data)
        if bias:
            b_data = cp.asarray(np.zeros(out_features, dtype=np.float32))
            self.bias = Param(b_data)
        else:
            self.bias = None
        self.params = [self.weight]
        if self.bias is not None:
            self.params.append(self.bias)
    def forward(self, x):
        self.x = x
        out = cp.dot(x, self.weight.data)
        if self.bias is not None:
            out += self.bias.data
        return out
    def backward(self, dout):
        self.weight.grad = cp.dot(self.x.T, dout)
        if self.bias is not None:
            self.bias.grad = cp.sum(dout, axis=0)
        return cp.dot(dout, self.weight.data.T)

class Dropout:
    def __init__(self, p=0.5, active=True):
        self.p = p
        self.active = active
    def forward(self, x, training=True):
        if not self.active or not training:
            self.mask = None
            return x
        self.mask = cp.random.binomial(1, 1 - self.p, size=x.shape).astype(cp.float32)
        return x * self.mask / (1 - self.p)
    def backward(self, dout):
        if self.mask is None:
            return dout
        return dout * self.mask / (1 - self.p)

class GlobalAvgPool2D:
    def forward(self, x):
        self.x_shape = x.shape
        return cp.mean(x, axis=(1, 2), keepdims=False)
    def backward(self, dout):
        N, H, W, C = self.x_shape
        dx = dout[:, None, None, :] / (H * W)
        return cp.broadcast_to(dx, self.x_shape)

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    def forward(self, x):
        N, H, W, C = x.shape
        k, s = self.kernel_size, self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        x_ = x.transpose(0, 3, 1, 2)
        col = im2col(x_, k, s, padding=0)
        col = col.reshape(-1, C, k*k)
        self.argmax = cp.argmax(col, axis=2)
        out = cp.max(col, axis=2)
        self.x_shape = x.shape
        self.col = col
        out = out.reshape(N, out_h, out_w, C)
        return out
    def backward(self, dout):
        N, H, W, C = self.x_shape
        k, s = self.kernel_size, self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        dout_flat = dout.reshape(-1, C)
        dcol = cp.zeros_like(self.col)
        batch_size = dout_flat.shape[0]
        for i in range(batch_size):
            for c in range(C):
                dcol[i, c, self.argmax[i, c]] = dout_flat[i, c]
        dcol = dcol.reshape(-1, C * k * k)
        dx_col = col2im(dcol, (N, C, H, W), k, s, 0)
        dx = dx_col.transpose(0, 2, 3, 1)
        return dx

class AvgPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    def forward(self, x):
        self.x_shape = x.shape
        N, H, W, C = x.shape
        k, s = self.kernel_size, self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        out = cp.zeros((N, out_h, out_w, C), dtype=x.dtype)
        self.out_h, self.out_w = out_h, out_w
        for i in range(out_h):
            for j in range(out_w):
                window = x[:, i*s:i*s+k, j*s:j*s+k, :]
                out[:, i, j, :] = cp.mean(window, axis=(1, 2))
        return out
    def backward(self, dout):
        N, H, W, C = self.x_shape
        k, s = self.kernel_size, self.stride
        dx = cp.zeros(self.x_shape, dtype=dout.dtype)
        for i in range(self.out_h):
            for j in range(self.out_w):
                grad = dout[:, i, j, :] / (k * k)
                dx[:, i*s:i*s+k, j*s:j*s+k, :] += grad[:, None, None, :]
        return dx

