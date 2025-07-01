# sgd.py
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [0 for _ in params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            if self.weight_decay != 0:
                param.grad += self.weight_decay * param.data
            v = self.momentum * self.velocities[i] - self.lr * param.grad
            self.velocities[i] = v
            param.data += v
