# cross_entropy.py
import cupy as cp

def logsumexp(x, axis=None, keepdims=False):
    c = cp.max(x, axis=axis, keepdims=True)
    return c + cp.log(cp.sum(cp.exp(x - c), axis=axis, keepdims=keepdims))

class CrossEntropyLoss:
    def forward(self, logits, labels):
        # logits: (N, C), labels: (N,) 정수형
        self.logits = logits
        self.labels = labels
        lse = logsumexp(logits, axis=1, keepdims=True)
        log_probs = logits - lse
        self.probs = cp.exp(log_probs)
        N = logits.shape[0]
        loss = -cp.mean(log_probs[cp.arange(N), labels])
        return loss

    def backward(self):
        N = self.logits.shape[0]
        grad = self.probs.copy()
        grad[cp.arange(N), self.labels] -= 1
        return grad / N

