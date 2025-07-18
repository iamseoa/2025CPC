import os
import tarfile
import urllib.request
import numpy as np
import cupy as cp
import pickle
from PIL import Image

np.random.seed(42)

# ------------------- Augmentation --------------------

def random_crop_flip(img, padding=4, flip_prob=0.5):
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w = img.shape[:2]
    top = np.random.randint(0, h - 32 + 1)
    left = np.random.randint(0, w - 32 + 1)
    cropped = img[top:top+32, left:left+32]
    if np.random.rand() < flip_prob:
        cropped = np.fliplr(cropped)
    return cropped

def cutout(img, mask_size=6):
    h, w, _ = img.shape
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    img[y1:y2, x1:x2, :] = 0.0
    return img

def apply_augmentation(img, aug_type="basic"):
    img_aug = img
    if aug_type in {"basic", "randaug", "cutout"}:
        img_aug = random_crop_flip(img_aug)
    if aug_type == "cutout":
        img_aug = cutout(img_aug)
    return img_aug

# ------------------- Dataloader (online aug) --------------------

class DataLoader:
    def __init__(self, x, y, batch_size=128, shuffle=True, augmentation="none"):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
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

        if self.augmentation != "none":
            batch_x = cp.asnumpy(batch_x)
            batch_x = np.stack([apply_augmentation(img, self.augmentation) for img in batch_x])
            batch_x = cp.asarray(batch_x)

        return batch_x, batch_y

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    @property
    def dataset(self):
        return self

# ------------------- Dataset Loader --------------------

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
        return pickle.load(fo, encoding='bytes')

def _preprocess(x):
    x = x.astype(np.float32) / 255.0
    x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x

def _to_gpu(*arrays):
    return [cp.asarray(a) for a in arrays]

def load_cifar100(batch_size=128, augmentation="none", split_ratio=0.8, seed=42):
    download_and_extract_cifar100()
    train_dict = _unpickle('cifar-100-python/train')
    test_dict = _unpickle('cifar-100-python/test')

    x_train = _preprocess(train_dict[b'data'])
    y_train = np.array(train_dict[b'fine_labels'])

    np.random.seed(seed)
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    split = int(len(x_train) * split_ratio)
    x_tr, y_tr = x_train[:split], y_train[:split]
    x_val, y_val = x_train[split:], y_train[split:]

    x_test = _preprocess(test_dict[b'data'])
    y_test = np.array(test_dict[b'fine_labels'])

    x_tr, x_val, x_test = _to_gpu(x_tr, x_val, x_test)
    y_tr, y_val, y_test = _to_gpu(y_tr, y_val, y_test)

    train_loader = DataLoader(x_tr, y_tr, batch_size=batch_size, shuffle=True, augmentation=augmentation)
    val_loader = DataLoader(x_val, y_val, batch_size=batch_size, shuffle=False, augmentation="none")
    test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False, augmentation="none")

    print("CIFAR-100 Loaded:")
    print(" x_train:", x_tr.shape)
    print(" y_train:", y_tr.shape)
    print(" x_val:", x_val.shape)
    print(" y_val:", y_val.shape)
    print(" x_test:", x_test.shape)
    print(" y_test:", y_test.shape)

    return train_loader, val_loader, test_loader

