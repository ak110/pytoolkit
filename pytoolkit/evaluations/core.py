"""評価。

指標名と値(float or ndarray)のdictを基本としてそれを扱うヘルパーなどをここで提供する。

"""
import typing
import numbers

import numpy as np

# 評価指標と値の型
EvalsType = typing.Dict[str, typing.Union[numbers.Number, np.ndarray]]


def to_str(evals: EvalsType) -> str:
    """文字列化。"""
    max_len = max(len(k) for k in evals) if len(evals) > 0 else 0
    s = [_to_str_kv(k, v, max_len) for k, v in evals.items()]
    return "\n".join(s)


def _to_str_kv(k, v, max_len) -> str:
    if isinstance(v, numbers.Number):
        return f"{k}:{' ' * (max_len - len(k))} {v:.3f}"
    elif isinstance(v, np.ndarray):
        s = np.array_str(v, precision=3, suppress_small=True)
        return f"{k}:{' ' * (max_len - len(k))} {s}\n"
    else:
        return f"{k}:{' ' * (max_len - len(k))} {v}\n"


def mean(
    evals_list: typing.Sequence[EvalsType], weights: typing.Sequence[float] = None
) -> EvalsType:
    """複数のevalsの要素ごとの平均を取る。キーは一致している前提。"""
    if weights is not None:
        assert len(evals_list) == len(weights)
    assert len(evals_list) > 0
    return {
        k: np.average([evals[k] for evals in evals_list], axis=0, weights=weights)
        for k in evals_list[0]
    }
