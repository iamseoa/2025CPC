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

def run_experiment(model_class, model_name, batch_size, lr, epochs, patience, momentum, dropout_p, resume_path=None, test_only=False):
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)

    hp_info = f"bs{batch_size}_lr{lr}_mom{momentum}_drop{dropout_p}"
    save_name = f"{model_name}_{hp_info}"

    if model_name == "MobileNetV2":
        model = model_class(num_classes=100, dropout_p=dropout_p)
    else:
        model = model_class(num_classes=100)
    print_model_size(model.params, save_name)
    optimizer = SGD(model.params, lr=lr, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader,
                      max_epochs=epochs, patience=patience,
                      experiment_name=f"hyperparam_{save_name}", model_name=save_name)

    if test_only and resume_path:
        assign_weights(model, load_weights(resume_path))
        top1, top5 = trainer.evaluate(test_loader, save_cm=True)
        print(f"Test Accuracy for {save_name} (loaded weights): Top1={top1:.4f}, Top5={top5:.4f}")
        return

    trainer.train(resume_path=resume_path)
    top1, top5 = trainer.evaluate(test_loader, save_cm=True)
    print(f"Test Accuracy for {save_name}: Top1={top1:.4f}, Top5={top5:.4f}")

if __name__ == "__main__":
    cp.random.seed(42)

    batch_sizes = [64, 128]
    learning_rates = [0.01, 0.001]
    momentums = [0.9]
    dropout_ps = [0.2, 0.3]
    epochs = 100
    patience = 10

    models = [
        (MiniDenseNet, "MiniDenseNet"),
        (MobileNetV2, "MobileNetV2")
    ]

    for model_class, model_name in models:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for momentum in momentums:
                    for dropout_p in dropout_ps:
                        print(f"Training {model_name} with batch_size={batch_size}, lr={lr}, momentum={momentum}, dropout={dropout_p}")
                        run_experiment(model_class, model_name, batch_size, lr, epochs, patience, momentum, dropout_p)
