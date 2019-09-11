import numpy as np

import pytoolkit as tk


def test_print_classification_metrics_multi():
    y_true = np.array([0, 1, 1, 1, 2])
    prob_pred = np.array(
        [
            [0.75, 0.00, 0.25],
            [0.25, 0.75, 0.00],
            [0.25, 0.75, 0.00],
            [0.25, 0.00, 0.75],
            [0.25, 0.75, 0.00],
        ]
    )
    tk.evaluations.print_classification_metrics(y_true, prob_pred)


def test_print_classification_metrics_binary():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.25, 0.25, 0.75, 0.25])
    tk.evaluations.print_classification_metrics(y_true, prob_pred)


def test_print_classification_metrics_binary_multi():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([[0.25, 0.75], [0.25, 0.75], [0.75, 0.25], [0.25, 0.75]])
    tk.evaluations.print_classification_metrics(y_true, prob_pred)


def test_print_regression_metrics():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.25, 0.25, 0.75, 0.25])
    tk.evaluations.print_regression_metrics(y_true, prob_pred)
