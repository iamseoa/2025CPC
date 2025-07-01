# baseline_vgg11.py
import os
import cupy as cp
from dataset import load_cifar100
from models.CustomVGGNet import CustomVGGNet as VGG11
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD
from utils.train_eval import Trainer
from utils.params_count import print_model_size

def run_experiment(model_class, model_name, batch_size=128, lr=0.01, epochs=5, patience=10):
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)
    model = model_class(num_classes=100)
    print_model_size(model.params, model_name)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params, lr=lr)
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader,
                      max_epochs=epochs, patience=patience,
                      experiment_name="baseline", model_name=model_name)
    trainer.train()
    top1, cm = trainer.evaluate(test_loader, save_cm=True)
    trainer._write_log(f"Test Top-1 Accuracy: {top1:.4f}")

if __name__ == "__main__":
    cp.random.seed(42)
    run_experiment(VGG11, "VGG11")

