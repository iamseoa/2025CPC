import os
import tarfile
import urllib.request
import numpy as np
import cupy as cp
import pickle
from PIL import Image, ImageOps, ImageEnhance

np.random.seed(42)

# --------------------------- Augmentation ---------------------------

def random_crop_flip(img, padding=2, flip_prob=0.3):
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w = img.shape[:2]
    top = np.random.randint(0, h - 32 + 1)
    left = np.random.randint(0, w - 32 + 1)
    cropped = img[top:top+32, left:left+32]
    if np.random.rand() < flip_prob:
        cropped = np.fliplr(cropped)
    return cropped

def randaug(img):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = ImageEnhance.Color(pil).enhance(0.9 + 0.2 * np.random.rand())      # [0.9 ~ 1.1]
    pil = ImageEnhance.Brightness(pil).enhance(0.9 + 0.2 * np.random.rand()) # [0.9 ~ 1.1]
    return np.asarray(pil).astype(np.float32) / 255.0

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

def color_jitter(img):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = ImageEnhance.Brightness(pil).enhance(0.95 + 0.1 * np.random.rand())
    pil = ImageEnhance.Contrast(pil).enhance(0.95 + 0.1 * np.random.rand())
    pil = ImageEnhance.Color(pil).enhance(0.95 + 0.1 * np.random.rand())
    return np.asarray(pil).astype(np.float32) / 255.0

def gaussian_noise(img, std=0.05):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)

def apply_augmentation(x, aug_type="basic"):
    out = []
    for img in x:
        if aug_type == "basic":
            img_aug = random_crop_flip(img)
        elif aug_type == "randaug":
            img_aug = random_crop_flip(img)
            img_aug = randaug(img_aug)
        elif aug_type == "cutout":
            img_aug = random_crop_flip(img)
            img_aug = cutout(img_aug)
        elif aug_type == "jitter":
            img_aug = random_crop_flip(img)
            img_aug = color_jitter(img_aug)
        elif aug_type == "gauss":
            img_aug = random_crop_flip(img)
            img_aug = gaussian_noise(img_aug)
        else:
            img_aug = img
        out.append(img_aug)
    return np.stack(out)



# ---------------------------- Dataloader ----------------------------

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

# ---------------------------- CIFAR Loader ---------------------------

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
    x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NCHW → NHWC
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

def load_cifar100(batch_size=128, augmentation="none"):
    download_and_extract_cifar100()
    train_dict = _unpickle('cifar-100-python/train')
    test_dict = _unpickle('cifar-100-python/test')

    x = np.concatenate([train_dict[b'data'], test_dict[b'data']], axis=0)
    y = np.concatenate([train_dict[b'fine_labels'], test_dict[b'fine_labels']], axis=0)
    x = _preprocess(x)
    x_train, y_train, x_val, y_val, x_test, y_test = _split_dataset(x, y)

    if augmentation != "none":
        x_train = apply_augmentation(x_train, aug_type=augmentation)

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

