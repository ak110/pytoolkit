import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_top_k_accuracy():
    y_true = np.array([1, 1, 1])
    proba_pred = np.array([[0.2, 0.1, 0.3], [0.1, 0.2, 0.3], [0.1, 0.3, 0.2]])
    assert tk.ml.top_k_accuracy(y_true, proba_pred, k=2) == pytest.approx(2 / 3)


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
    tk.ml.print_classification_metrics(y_true, prob_pred)


def test_print_classification_metrics_binary():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.25, 0.25, 0.75, 0.25])
    tk.ml.print_classification_metrics(y_true, prob_pred)


def test_print_classification_metrics_binary_multi():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([[0.25, 0.75], [0.25, 0.75], [0.75, 0.25], [0.25, 0.75]])
    tk.ml.print_classification_metrics(y_true, prob_pred)


def test_print_regression_metrics():
    y_true = np.array([0, 1, 1, 0])
    prob_pred = np.array([0.25, 0.25, 0.75, 0.25])
    tk.ml.print_regression_metrics(y_true, prob_pred)
