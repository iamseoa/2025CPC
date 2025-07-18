import os
import cupy as cp
import numpy as np
from dataset import load_cifar100
from models.CustomMobileNet import CustomMobileNet as MobileNetV2
from models.CustomResNet import CustomResNet as ResNet20
from models.CustomDenseNet import CustomDenseNet as MiniDenseNet
from losses.cross_entropy import CrossEntropyLoss
from losses.label_smoothing_ce import LabelSmoothingCrossEntropy
from optim.sgd import SGD
from utils.train_eval import Trainer
from utils.params_count import print_model_size

def safe_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray, cp.ndarray)):
        x = np.asarray(x)
        if x.size == 1:
            return float(x.item())
        return float(np.mean(x))
    return float(x)

def run_experiment(model_class, model_name, batch_size, lr, epochs, patience,
                   momentum, dropout_p, label_smoothing):
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)
    save_name = f"{model_name}_smooth{label_smoothing}"

    if model_name == "MobileNetV2":
        model = model_class(num_classes=100, dropout_p=dropout_p)
    else:
        model = model_class(num_classes=100)

    print_model_size(model.params, save_name)

    optimizer = SGD(model.params, lr=lr, momentum=momentum, weight_decay=0.0)

    if label_smoothing > 0.0:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing, num_classes=100)
    else:
        loss_fn = CrossEntropyLoss()

    trainer = Trainer(
        model, optimizer, loss_fn, train_loader, val_loader,
        max_epochs=epochs, patience=patience,
        scheduler=None,
        experiment_name=f"labelsmooth_{model_name}",
        model_name=save_name
    )
    trainer.train()
    top1, top5 = trainer.evaluate(test_loader, save_cm=True)
    print(f"Test Accuracy for {save_name}: Top1={safe_scalar(top1):.4f}, Top5={safe_scalar(top5):.4f}")

if __name__ == "__main__":
    cp.random.seed(42)
    epochs = 100
    patience = 10
    smoothing_values = [0.1, 0.15, 0.2]

    model_settings = [
        (ResNet20, "ResNet20", 64, 0.05, 0.99, 0.2),
        (MiniDenseNet, "MiniDenseNet", 128, 0.005, 0.9, 0.4),
        (MobileNetV2, "MobileNetV2", 128, 0.005, 0.9, 0.5),
    ]

    for model_class, model_name, batch_size, lr, momentum, dropout_p in model_settings:
        for smoothing in smoothing_values:
            print(f"Training {model_name} with label_smoothing={smoothing}")
            run_experiment(
                model_class=model_class,
                model_name=model_name,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                patience=patience,
                momentum=momentum,
                dropout_p=dropout_p,
                label_smoothing=smoothing
            )

