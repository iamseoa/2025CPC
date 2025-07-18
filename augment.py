import os
import cupy as cp
import numpy as np
from dataset import load_cifar100
from models.CustomMobileNet import CustomMobileNet as MobileNetV2
from models.CustomResNet import CustomResNet as ResNet20
from models.CustomDenseNet import CustomDenseNet as MiniDenseNet
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD
from optim.adam import Adam
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

def safe_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray, cp.ndarray)):
        x = np.asarray(x)
        if x.size == 1:
            return float(x.item())
        return float(np.mean(x))
    return float(x)

def run_experiment(model_class, model_name, batch_size, lr, epochs, patience,
                   momentum, dropout_p, scheduler_name, aug_type, optimizer_name):

    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size, augmentation=aug_type)
    save_name = f"{model_name}_{aug_type}_{optimizer_name}"

    if model_name == "MobileNetV2":
        model = model_class(num_classes=100, dropout_p=dropout_p)
    else:
        model = model_class(num_classes=100)

    print_model_size(model.params, save_name)

    if optimizer_name == "sgd":
        optimizer = SGD(model.params, lr=lr, momentum=momentum, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = Adam(model.params, lr=lr, weight_decay=1e-4)
    else:
        raise ValueError("Unknown optimizer")

    if scheduler_name == "steplr":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError("Unknown scheduler")

    loss_fn = CrossEntropyLoss()

    trainer = Trainer(
        model, optimizer, loss_fn, train_loader, val_loader,
        max_epochs=epochs, patience=patience,
        scheduler=scheduler,
        experiment_name=f"augexp_{model_name}",
        model_name=save_name
    )
    trainer.train()
    top1, top5 = trainer.evaluate(test_loader, save_cm=True)
    print(f"Test Accuracy for {save_name}: Top1={safe_scalar(top1):.4f}, Top5={safe_scalar(top5):.4f}")

if __name__ == "__main__":
    cp.random.seed(42)
    epochs = 100
    patience = 10

    aug_types = ["basic", "randaug", "cutout"]

    fixed_settings = {
        "ResNet20":     {"batch_size": 64,  "lr": 0.01,  "momentum": 0.99, "dropout": 0.2, "scheduler": "cosine",  "optimizer": "sgd"},
        "MiniDenseNet": {"batch_size": 128, "lr": 0.01,  "momentum": 0.9,  "dropout": 0.4, "scheduler": "plateau", "optimizer": "sgd"},
        "MobileNetV2":  {"batch_size": 128, "lr": 0.001, "momentum": 0.0,  "dropout": 0.3, "scheduler": "plateau",  "optimizer": "adam"},
    }

    model_classes = {
        "ResNet20": ResNet20,
        "MiniDenseNet": MiniDenseNet,
        "MobileNetV2": MobileNetV2
    }

    for model_name in ["ResNet20", "MiniDenseNet", "MobileNetV2"]:
        config = fixed_settings[model_name]
        for aug_type in aug_types:
            print(f"Training {model_name} | aug={aug_type} | opt={config['optimizer']}")
            run_experiment(
                model_class=model_classes[model_name],
                model_name=model_name,
                batch_size=config["batch_size"],
                lr=config["lr"],
                epochs=epochs,
                patience=patience,
                momentum=config["momentum"],
                dropout_p=config["dropout"],
                scheduler_name=config["scheduler"],
                aug_type=aug_type,
                optimizer_name=config["optimizer"]
            )

