import cupy as cp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from utils.metrics import accuracy_topk, confusion_matrix

def save_weights(path, weights):
    np.savez(path, **{f'param_{i}': w.get() for i, w in enumerate(weights)})

def load_weights(path):
    d = np.load(path)
    return [d[f'param_{i}'] for i in range(len(d.files))]

def assign_weights(model, weight_list):
    for param, w in zip(model.params, weight_list):
        param.data[...] = cp.asarray(w)

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader,
                 max_epochs=200, patience=20, experiment_name="exp", model_name="model"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.experiment_name = experiment_name
        self.model_name = model_name

        self.train_losses = []
        self.val_losses = []
        self.best_weights = None
        self.best_val_acc = 0
        self.counter = 0

        self.output_dir = os.path.join("logs", f"{experiment_name}_{model_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.cm_file = os.path.join(self.output_dir, "confusion_matrix.png")
        self.loss_curve_file = os.path.join(self.output_dir, "loss_curve.png")

    def train(self, resume_path=None):
        if resume_path is not None:
            self._write_log(f"Resume training from: {resume_path}")
            loaded_weights = load_weights(resume_path)
            assign_weights(self.model, loaded_weights)
        total_start = time.time()
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)

            self.train_losses.append(float(train_loss))
            self.val_losses.append(float(val_loss))

            val_acc, _ = self.evaluate(self.val_loader, save_cm=False)
            epoch_time = time.time() - epoch_start

            log_line = (f"Epoch {epoch+1}: Train Loss={float(train_loss):.4f}, "
                        f"Val Loss={float(val_loss):.4f}, Val Top-1 Acc={val_acc:.4f}, "
                        f"Epoch Time={epoch_time:.2f}s")
            self._write_log(log_line)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_weights = self._get_model_weights()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self._write_log(f"Early stopping at epoch {epoch+1}")
                    break

        total_time = time.time() - total_start
        self._set_model_weights(self.best_weights)
        self.plot_loss_curve()
        _, _ = self.evaluate(self.val_loader, save_cm=True)

        checkpoint_path = os.path.join("checkpoints", f"{self.experiment_name}_{self.model_name}.npz")
        save_weights(checkpoint_path, self.best_weights)
        self._write_log(f"Saved best model weights to {checkpoint_path}")
        self._write_log(f"Total training time: {total_time:.2f}s")

    def _run_epoch(self, loader, training):
        total_loss = 0
        total_samples = 0
        for x, y in loader:
            logits = self.model.forward(x)
            loss = self.loss_fn.forward(logits, y)
            batch_size = x.shape[0]
            if training:
                grad = self.loss_fn.backward()
                self._backward(grad)
                self.optimizer.step()
            total_loss += float(loss) * batch_size
            total_samples += batch_size
        return total_loss / total_samples

    def evaluate(self, loader, save_cm=False, weight_path=None, return_logits=False):
        if weight_path is not None:
            loaded_weights = load_weights(weight_path)
            assign_weights(self.model, loaded_weights)
        all_preds = []
        all_labels = []
        for x, y in loader:
            logits = self.model.forward(x)
            all_preds.append(cp.asnumpy(logits))
            all_labels.append(cp.asnumpy(y))
        logits = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        top1, top5 = accuracy_topk(logits, labels)
        cm = confusion_matrix(logits, labels, topk=1, save_path=self.cm_file if save_cm else None)
        if return_logits:
            return top1, cm, logits, labels
        return top1, cm

    def _backward(self, grad):
        for layer in reversed(self.model.layers):
            grad = layer.backward(grad)

    def _get_model_weights(self):
        return [cp.copy(param.data) for param in self.model.params]

    def _set_model_weights(self, weights):
        for param, w in zip(self.model.params, weights):
            param.data[...] = w

    def _write_log(self, text):
        print(text)
        with open(self.log_file, 'a') as f:
            f.write(text + "\n")

    def plot_loss_curve(self):
        plt.figure()
        plt.plot(np.array(self.train_losses), label='Train')
        plt.plot(np.array(self.val_losses), label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_curve_file)
        plt.close()

