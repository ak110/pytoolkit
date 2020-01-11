import numpy as np
import pytest

import pytoolkit as tk


def test_to_str():
    evals = {"aaa": 1, "b": np.float32(2)}

    s = tk.evaluations.to_str(evals)
    assert s == "aaa=1.000 b=2.000"

    s = tk.evaluations.to_str(evals, multiline=True)
    assert s == "aaa: 1.000\nb:   2.000"

    s = tk.evaluations.to_str({"a": ErrorType()})
    assert (
        s == "a=<class 'pytoolkit.evaluations.core_test.ErrorType'>"
        or s == "a=<class 'pytoolkit.pytoolkit.evaluations.core_test.ErrorType'>"
    )


def test_mean():
    evals1 = {"a": 1, "b": 1}
    evals2 = {"a": 0, "b": 0.5}
    assert tk.evaluations.mean([evals1, evals2]) == pytest.approx({"a": 0.5, "b": 0.75})


class ErrorType:
    """str()出来ないクラス。"""

    def __str__(self):
        return 0
