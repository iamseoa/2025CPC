import os
import cupy as cp
from dataset import load_cifar100
from models.CustomResNet import CustomResNet as ResNet20
from models.CustomDenseNet import CustomDenseNet as MiniDenseNet
from models.CustomVGGNet import CustomVGGNet as VGG11
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD
from utils.train_eval import Trainer, load_weights, assign_weights
from utils.params_count import print_model_size

def check_cupy_device():
    device = cp.cuda.Device()
    print(f"Cupy GPU Device: {device.id} - {cp.cuda.runtime.getDeviceProperties(device.id)['name']}")
    print(f"Memory total: {cp.cuda.runtime.memGetInfo()[1] // (1024 ** 2)} MB")
    print(f"Memory free : {cp.cuda.runtime.memGetInfo()[0] // (1024 ** 2)} MB")
    print(f"Cupy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")

def run_experiment(model_class, model_name, batch_size=128, lr=0.01, epochs=100, patience=5, resume_path=None, test_only=False):
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)
    model = model_class(num_classes=100)
    print_model_size(model.params, model_name)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params, lr=lr)
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader,
                      max_epochs=epochs, patience=patience,
                      experiment_name="baseline_epoch100", model_name=model_name)

    if test_only and resume_path is not None:
        assign_weights(model, load_weights(resume_path))
        top1, top5 = trainer.evaluate(test_loader, save_cm=True)
        trainer._write_log(f"Test Top-1 Accuracy (loaded weights): {top1:.4f}")
        trainer._write_log(f"Test Top-5 Accuracy (loaded weights): {top5:.4f}")
        return

    trainer.train(resume_path=resume_path)
    top1, top5 = trainer.evaluate(test_loader, save_cm=True)
    trainer._write_log(f"Test Top-1 Accuracy: {top1:.4f}")
    trainer._write_log(f"Test Top-5 Accuracy: {top5:.4f}")

if __name__ == "__main__":
    check_cupy_device()
    cp.random.seed(42)
    batch_size = 128
    lr = 0.01
    epochs = 100
    patience = 5  

    # ResNet20
    run_experiment(ResNet20, "ResNet20", batch_size=batch_size, lr=lr, epochs=epochs, patience=patience)
    run_experiment(ResNet20, "ResNet20", test_only=True, resume_path="checkpoints/baseline_epoch100_ResNet20.npz")

    # DenseNet
    run_experiment(MiniDenseNet, "MiniDenseNet", batch_size=batch_size, lr=lr, epochs=epochs, patience=patience)
    run_experiment(MiniDenseNet, "MiniDenseNet", test_only=True, resume_path="checkpoints/baseline_epoch100_MiniDenseNet.npz")

    # VGG11
    run_experiment(VGG11, "VGG11", batch_size=batch_size, lr=lr, epochs=epochs, patience=patience)
    run_experiment(VGG11, "VGG11", test_only=True, resume_path="checkpoints/baseline_epoch100_VGG11.npz")
