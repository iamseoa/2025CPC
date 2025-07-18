import os
import cupy as cp
from dataset import load_cifar100
from models.CustomMobileNet import CustomMobileNet as MobileNetV2
from models.CustomResNet import CustomResNet as ResNet20
from models.CustomDenseNet import CustomDenseNet as MiniDenseNet
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD
from utils.train_eval import Trainer, load_weights, assign_weights
from utils.params_count import print_model_size

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

def run_experiment(model_class, model_name, batch_size, lr, epochs, patience, momentum, dropout_p, init_method, init_dist):
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)

    hp_info = f"bs{batch_size}_lr{lr}_mom{momentum}_drop{dropout_p}"
    init_info = f"{init_method}-{init_dist}"
    save_name = f"{model_name}_{hp_info}_{init_info}"

    if model_name == "MobileNetV2":
        model = model_class(num_classes=100, dropout_p=dropout_p)
    else:
        model = model_class(num_classes=100)

    custom_initialize(model, init_method, init_dist)
    print_model_size(model.params, save_name)

    optimizer = SGD(model.params, lr=lr, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(
        model, optimizer, loss_fn, train_loader, val_loader,
        max_epochs=epochs, patience=patience,
        experiment_name=f"initcompare_{save_name}",
        model_name=save_name
    )

    trainer.train()
    top1, top5 = trainer.evaluate(test_loader, save_cm=True)
    print(f"Test Accuracy for {save_name}: Top1={top1:.4f}, Top5={top5:.4f}")

if __name__ == "__main__":
    cp.random.seed(42)

    init_methods = ["xavier", "he"]
    init_dists = ["uniform", "normal"]

    epochs = 100
    patience = 10

    model_settings = [
        (ResNet20, "ResNet20", 64, 0.01, 0.99, 0.2),
        (MiniDenseNet, "MiniDenseNet", 128, 0.001, 0.9, 0.2),
        (MobileNetV2, "MobileNetV2", 128, 0.001, 0.9, 0.3),
    ]

    for model_class, model_name, batch_size, lr, momentum, dropout_p in model_settings:
        for method in init_methods:
            for dist in init_dists:
                print(f"Training {model_name} with init={method}-{dist}")
                run_experiment(model_class, model_name, batch_size, lr, epochs, patience, momentum, dropout_p, method, dist)

