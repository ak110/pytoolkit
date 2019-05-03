"""機械学習関連。"""

import numpy as np


def sigmoid(x):
    """シグモイド関数。"""
    return 1 / (1 + np.exp(-x))


def logit(x, epsilon=1e-7):
    """シグモイド関数の逆関数。"""
    x = np.clip(x, epsilon, 1 - epsilon)
    return np.log(x / (1 - x))
