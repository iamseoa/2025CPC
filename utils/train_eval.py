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
                 max_epochs=200, patience=20, experiment_name="exp", model_name="model",
                 scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.train_top1s = []
        self.train_top5s = []
        self.val_top1s = []
        self.val_top5s = []
        self.best_weights = None
        self.best_val_acc = 0
        self.counter = 0

        self.output_dir = os.path.join("logs", f"{experiment_name}_{model_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.cm_file = os.path.join(self.output_dir, "confusion_matrix.png")
        self.loss_curve_file = os.path.join(self.output_dir, "loss_curve.png")
        self.acc_curve_file = os.path.join(self.output_dir, "acc_curve.png")

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

            train_top1, train_top5 = self.evaluate(self.train_loader, save_cm=False, calc_acc_only=True)
            val_top1, val_top5 = self.evaluate(self.val_loader, save_cm=False, calc_acc_only=True)

            self.train_losses.append(float(train_loss))
            self.val_losses.append(float(val_loss))
            self.train_top1s.append(train_top1)
            self.train_top5s.append(train_top5)
            self.val_top1s.append(val_top1)
            self.val_top5s.append(val_top5)

            epoch_time = time.time() - epoch_start

            log_line = (
                f"Epoch {epoch+1}: "
                f"Train Loss={float(train_loss):.4f}, "
                f"Train Top-1 Acc={float(train_top1):.4f}, Train Top-5 Acc={float(train_top5):.4f}, "
                f"Val Loss={float(val_loss):.4f}, "
                f"Val Top-1 Acc={float(val_top1):.4f}, Val Top-5 Acc={float(val_top5):.4f}, "
                f"Epoch Time={epoch_time:.2f}s"
            )
            self._write_log(log_line)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if "val_loss" in self.scheduler.step.__code__.co_varnames:
                        self.scheduler.step(float(val_loss))
                    else:
                        self.scheduler.step()

            if (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(
                    "checkpoints", f"{self.experiment_name}_{self.model_name}_epoch{epoch+1}.npz"
                )
                save_weights(ckpt_path, self._get_model_weights())
                self._write_log(f"Checkpoint saved at {ckpt_path}")

            if val_top1 > self.best_val_acc:
                self.best_val_acc = val_top1
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
        self.plot_acc_curve()
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

    def evaluate(self, loader, save_cm=False, weight_path=None, return_logits=False, calc_acc_only=False):
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
        if calc_acc_only:
            return top1, top5
        cm = confusion_matrix(logits, labels, topk=1, save_path=self.cm_file if save_cm else None)
        if return_logits:
            return top1, top5, cm, logits, labels
        return top1, top5

    def test(self, test_loader, save_cm=True):
        top1, top5 = self.evaluate(test_loader, save_cm=save_cm)
        self._write_log(f"Test Top-1 Accuracy: {top1:.4f}")
        self._write_log(f"Test Top-5 Accuracy: {top5:.4f}")
        return top1, top5

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
        plt.plot(np.array(self.train_losses), label='Train Loss')
        plt.plot(np.array(self.val_losses), label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_curve_file)
        plt.close()

    def plot_acc_curve(self):
        plt.figure()
        plt.plot(np.array(self.train_top1s), label='Train Top-1')
        plt.plot(np.array(self.val_top1s), label='Val Top-1')
        plt.plot(np.array(self.train_top5s), label='Train Top-5')
        plt.plot(np.array(self.val_top5s), label='Val Top-5')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.acc_curve_file)
        plt.close()

