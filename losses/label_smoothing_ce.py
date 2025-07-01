# label_smoothing_ce.py
import cupy as cp

def logsumexp(x, axis=None, keepdims=False):
    c = cp.max(x, axis=axis, keepdims=True)
    return c + cp.log(cp.sum(cp.exp(x - c), axis=axis, keepdims=keepdims))

class LabelSmoothingCrossEntropy:
    def __init__(self, smoothing=0.1, num_classes=100):
        self.eps = smoothing
        self.K = num_classes

    def forward(self, logits, labels):
        # logits: (N, C), labels: (N,) 정수형
        self.logits = logits
        self.labels = labels
        lse = logsumexp(logits, axis=1, keepdims=True)
        log_probs = logits - lse
        N = logits.shape[0]

        one_hot = cp.zeros_like(logits)
        one_hot[cp.arange(N), labels] = 1
        smoothed = (1 - self.eps) * one_hot + self.eps / self.K
        self.probs = cp.exp(log_probs)
        self.smoothed = smoothed

        loss = -cp.sum(smoothed * log_probs) / N
        return loss

    def backward(self):
        N = self.logits.shape[0]
        grad = self.probs - self.smoothed
        return grad / N

