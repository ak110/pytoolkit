import numpy as np

import pytoolkit as tk


def test_print_ss_binary():
    y_true = np.zeros((2, 32, 32))
    y_true[1, :, :] = 1
    y_pred = np.zeros((2, 32, 32))
    y_pred[:, :16, :16] = 1  # iou=0.25
    tk.evaluations.print_ss(y_true, y_pred)


def test_print_ss_multi():
    y_true = np.zeros((2, 32, 32, 3))
    y_true[1, :, :, :] = 1
    y_pred = np.zeros((2, 32, 32, 3))
    y_pred[:, :16, :16, :] = 1  # iou=0.25
    tk.evaluations.print_ss(y_true, y_pred)
