"""numpy関連。"""
import numbers

import numpy as np


def set_ndarray_format():
    """ndarrayのstr()やrepr()でshapeが分かるようにする。"""

    def format_ndarray(x):
        try:
            result = f"<ndarray shape={x.shape} dtype={x.dtype}"
            if issubclass(x.dtype.type, numbers.Number):
                result += f" min={x.min()}"
                result += f" max={x.max()}"
                result += f" mean={x.mean(dtype=np.float32)}"
            s = np.array_str(x).replace("\n", "")
            result += f" values={s}"
            result += f">"
            return result
        except BaseException:
            return np.array_repr(x)  # 念のため

    np.set_string_function(format_ndarray, repr=False)
    np.set_string_function(format_ndarray, repr=True)


def sigmoid(x):
    """シグモイド関数。"""
    return 1 / (1 + np.exp(-x))


def logit(x, epsilon=1e-7):
    """シグモイド関数の逆関数。"""
    x = np.clip(x, epsilon, 1 - epsilon)
    return np.log(x / (1 - x))
