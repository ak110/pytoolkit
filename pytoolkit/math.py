"""numpy関連。"""
import numbers
import typing

import numpy as np
import scipy


def set_ndarray_format() -> None:
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


def softmax(x, axis=-1):
    """ソフトマックス関数。"""
    return scipy.special.softmax(x, axis=axis)


def cosine_most_similars(
    v1: np.ndarray, v2: np.ndarray, batch_size: int = 10000
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """v1の各ベクトルに対して、v2の中で最もコサイン類似度の大きいindexとその類似度を返す。

    Args:
        v1: 基準となるベクトル (shape=(N, C))
        v2: 探す対象のベクトル (shape=(M, C))
        batch_size: いくつのベクトルをまとめて処理するのか (メモリ使用量に影響)

    Returns:
        indexの配列と類似度の配列 (shape=(N,))

    """
    assert v1.ndim == 2
    assert v2.ndim == 2
    if v1 is v2:
        v1 = v2 = l2_normalize(v1.astype(np.float32))
    else:
        v1 = l2_normalize(v1.astype(np.float32))
        v2 = l2_normalize(v2.astype(np.float32))

    indices = np.empty((len(v1)), dtype=np.int64)
    similarities = np.empty((len(v1)), dtype=np.float32)

    for offset in range(0, len(v1), batch_size):
        v1_batch = v1[offset : offset + batch_size]
        sims = np.inner(v1_batch, v2)
        indices[offset : offset + len(sims)] = sims.argmax(axis=-1)
        similarities[offset : offset + len(sims)] = sims.max(axis=-1)

    return indices, similarities


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """ベクトルのL2正規化。"""
    return x / np.linalg.norm(x, axis=axis, keepdims=True)
