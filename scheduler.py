# scheduler.py

import os
import cupy as cp
import numpy as np
from dataset import load_cifar100
from models.CustomMobileNet import CustomMobileNet as MobileNetV2
from models.CustomResNet import CustomResNet as ResNet20
from models.CustomDenseNet import CustomDenseNet as MiniDenseNet
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD
from utils.train_eval import Trainer
from utils.params_count import print_model_size

class StepLR:
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
    def step(self, metrics=None):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self.last_epoch = 0
        self.initial_lr = optimizer.lr
    def step(self, metrics=None):
        self.last_epoch += 1
        cosine_decay = (1 + cp.cos(cp.pi * self.last_epoch / self.T_max)) / 2
        self.optimizer.lr = self.initial_lr * cosine_decay

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.5, patience=5, verbose=False, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.best = float('inf')
        self.num_bad_epochs = 0
    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            if self.verbose:
                print(f"ReduceLROnPlateau: Reducing learning rate to {new_lr:.6f}")
            self.optimizer.lr = new_lr
            self.num_bad_epochs = 0

def custom_initialize(model, method, dist):
    for param in model.params:
        if not hasattr(param, "shape") or not isinstance(param, cp.ndarray):
            continue
        shape = param.shape
        if len(shape) < 2:
            continue
        fan_in, fan_out = shape[1], shape[0]
        if method == "xavier":
            scale = cp.sqrt(6.0 / (fan_in + fan_out)) if dist == "uniform" else cp.sqrt(2.0 / (fan_in + fan_out))
        elif method == "he":
            scale = cp.sqrt(6.0 / fan_in) if dist == "uniform" else cp.sqrt(2.0 / fan_in)
        else:
            raise ValueError("Invalid init method")
        if dist == "uniform":
            param[...] = cp.random.uniform(-scale, scale, shape)
        elif dist == "normal":
            param[...] = cp.random.normal(0, scale, shape)

def safe_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray, cp.ndarray)):
        x = np.asarray(x)
        if x.size == 1:
            return float(x.item())
        return float(np.mean(x))
    return float(x)

def run_experiment(model_class, model_name, batch_size, lr, epochs, patience,
                   momentum, dropout_p, scheduler_name, init_method, init_dist):
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)
    save_name = f"{model_name}_bs{batch_size}_lr{lr}_mom{momentum}_drop{dropout_p}_{init_method}-{init_dist}_{scheduler_name}"
    if model_name == "MobileNetV2":
        model = model_class(num_classes=100, dropout_p=dropout_p)
    else:
        model = model_class(num_classes=100)
    custom_initialize(model, init_method, init_dist)
    print_model_size(model.params, save_name)
    optimizer = SGD(model.params, lr=lr, momentum=momentum)

    if scheduler_name == "steplr":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    else:
        raise ValueError("Unknown scheduler")

    loss_fn = CrossEntropyLoss()

    trainer = Trainer(
        model, optimizer, loss_fn, train_loader, val_loader,
        max_epochs=epochs, patience=patience,
        scheduler=scheduler,
        experiment_name=f"scheduler_{model_name}",
        model_name=save_name
    )
    trainer.train()
    top1, top5 = trainer.evaluate(test_loader, save_cm=True)
    print(f"Test Accuracy for {save_name}: Top1={safe_scalar(top1):.4f}, Top5={safe_scalar(top5):.4f}")

if __name__ == "__main__":
    cp.random.seed(42)
    epochs = 100
    patience = 10
    schedulers = ["steplr", "cosine", "plateau"]
    model_settings = [
        (ResNet20, "ResNet20", 64, 0.01, 0.99, 0.2, "xavier", "uniform"),
        (MiniDenseNet, "MiniDenseNet", 128, 0.001, 0.9, 0.2, "he", "normal"),
        (MobileNetV2, "MobileNetV2", 128, 0.001, 0.9, 0.3, "he", "normal"),
    ]
    for model_class, model_name, batch_size, lr, momentum, dropout_p, init_method, init_dist in model_settings:
        for scheduler_name in schedulers:
            print(f"Training {model_name} with LR scheduler: {scheduler_name}")
            run_experiment(model_class, model_name, batch_size, lr, epochs, patience,
                           momentum, dropout_p, scheduler_name, init_method, init_dist)

