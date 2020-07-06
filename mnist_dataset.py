import torch
import torchvision
import numpy as np


MNIST_NORM_MEAN = 0.1307
MNIST_NORM_STD = 0.3081


def get_train_dataloader(path="dataset", batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path,
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_NORM_MEAN,), (MNIST_NORM_STD,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    return train_loader


def get_test_dataloader(path="dataset", batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path,
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_NORM_MEAN,), (MNIST_NORM_STD,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    return train_loader


def denormalize_image(img):
    if type(img) == torch.Tensor:
        img = img.numpy()
    img = np.multiply(img, MNIST_NORM_STD)
    img = np.add(img, MNIST_NORM_MEAN)
    img = np.multiply(img, 255).astype(np.uint8)
    return img


def normalize_img_cyx(img):
    img = np.multiply(img, 1/255.0)
    img = np.subtract(img, MNIST_NORM_MEAN)
    img = np.divide(img, MNIST_NORM_STD)
    return img
