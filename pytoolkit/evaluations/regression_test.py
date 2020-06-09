import numpy as np

import pytoolkit as tk


def test_print_regression():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.25, 0.25, 0.75, 0.25])
    tk.evaluations.print_regression(y_true, prob_pred)
