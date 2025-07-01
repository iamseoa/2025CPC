# metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_topk(logits, labels, topk=(1, 5)):
    """
    logits: (N, C) numpy array
    labels: (N,) numpy array
    """
    maxk = max(topk)
    batch_size = labels.shape[0]

    pred = np.argsort(logits, axis=1)[:, ::-1][:, :maxk]  # (N, maxk)
    correct = pred == labels[:, None]  # (N, maxk)

    res = []
    for k in topk:
        correct_k = np.sum(correct[:, :k])
        res.append(correct_k / batch_size)
    return res if len(res) > 1 else res[0]

def confusion_matrix(logits, labels, topk=1, save_path=None):
    """
    logits: (N, C) numpy array
    labels: (N,) numpy array
    save_path: path to save the confusion matrix image (optional)
    """
    pred = np.argsort(logits, axis=1)[:, ::-1][:, :topk]  # (N, topk)
    if topk == 1:
        pred = pred.flatten()
    else:
        pred = pred[:, 0]
    cm = sk_confusion_matrix(labels, pred)

    if save_path:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return cm

