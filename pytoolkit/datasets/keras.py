"""keras.datasets関連。 <https://keras.io/datasets/>"""
import numpy as np

import pytoolkit as tk


def load_mnist():
    """MNIST."""
    (X_train, y_train), (X_test, y_test) = tk.keras.datasets.mnist.load_data()
    X_train = X_train.reshape((60000, 28, 28, 1))
    X_test = X_test.reshape((10000, 28, 28, 1))
    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)


def load_fashion_mnist():
    """Fashion-MNIST."""
    (X_train, y_train), (X_test, y_test) = tk.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape((60000, 28, 28, 1))
    X_test = X_test.reshape((10000, 28, 28, 1))
    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)


def load_cifar10():
    """CIFAR10."""
    (X_train, y_train), (X_test, y_test) = tk.keras.datasets.cifar10.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)


def load_cifar100():
    """CIFAR100."""
    (X_train, y_train), (X_test, y_test) = tk.keras.datasets.cifar100.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)
