# dataset.py
import os
import tarfile
import urllib.request
import numpy as np
import cupy as cp
import pickle

np.random.seed(42)

class DataLoader:
    def __init__(self, x, y, batch_size=128, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
        self.indices = np.arange(self.num_samples)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        self.current_idx += self.batch_size
        return batch_x, batch_y

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    @property
    def dataset(self):
        return self

def download_and_extract_cifar100(destination='cifar-100-python'):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    archive_name = 'cifar-100-python.tar.gz'
    if not os.path.exists(archive_name):
        urllib.request.urlretrieve(url, archive_name)
    if not os.path.exists(destination):
        with tarfile.open(archive_name, 'r:gz') as tar:
            tar.extractall()

def _unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def _preprocess(x):
    x = x.astype(np.float32) / 255.0
    x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x

def _split_dataset(x, y):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    return (
        x[:40000], y[:40000],
        x[40000:50000], y[40000:50000],
        x[50000:], y[50000:]
    )

def _to_gpu(*arrays):
    return [cp.asarray(a) for a in arrays]

def load_cifar100(batch_size=128):
    download_and_extract_cifar100()
    train_dict = _unpickle('cifar-100-python/train')
    test_dict = _unpickle('cifar-100-python/test')

    x = np.concatenate([train_dict[b'data'], test_dict[b'data']], axis=0)
    y = np.concatenate([train_dict[b'fine_labels'], test_dict[b'fine_labels']], axis=0)
    x = _preprocess(x)
    x_train, y_train, x_val, y_val, x_test, y_test = _split_dataset(x, y)

    x_train, y_train, x_val, y_val, x_test, y_test = _to_gpu(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)

    print("CIFAR-100 Loaded:")
    print(" x_train:", x_train.shape)
    print(" y_train:", y_train.shape)
    print(" x_val:", x_val.shape)
    print(" y_val:", y_val.shape)
    print(" x_test:", x_test.shape)
    print(" y_test:", y_test.shape)

    return train_loader, val_loader, test_loader

