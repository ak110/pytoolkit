import numpy as np
import pytest

import pytoolkit as tk


def test_to_str():
    evals = {"aaa": 1, "b": np.float32(2)}
    s = tk.evaluations.to_str(evals)
    assert s == "aaa: 1.000\nb:   2.000"


def test_mean():
    evals1 = {"a": 1, "b": 1}
    evals2 = {"a": 0, "b": 0.5}
    assert tk.evaluations.mean([evals1, evals2]) == pytest.approx({"a": 0.5, "b": 0.75})
