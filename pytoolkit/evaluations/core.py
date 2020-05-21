"""評価。

指標名と値(float or ndarray)のdictを基本としてそれを扱うヘルパーなどをここで提供する。

"""
import numbers
import typing

import numpy as np

# 評価指標と値の型
EvalsType = typing.Dict[str, typing.Any]


def to_str(evals: EvalsType, multiline=False, precision=3) -> str:
    """文字列化。"""
    max_len = max(len(k) for k in evals) if multiline and len(evals) > 0 else None
    s = [_to_str_kv(k, v, max_len, precision) for k, v in evals.items()]
    sep = "\n" if multiline else " "
    return sep.join(s)


def _to_str_kv(k, v, max_len, precision) -> str:
    if max_len is None:
        sep = "="
    else:
        assert max_len >= len(k)
        sep = ":" + " " * (max_len - len(k) + 1)
    try:
        if isinstance(v, numbers.Number):
            return f"{k}{sep}{v:,.{precision}f}"
        elif isinstance(v, np.ndarray):
            v = np.array_str(v, precision=precision, suppress_small=True)
            return f"{k}{sep}{v}"
        else:
            return f"{k}{sep}{v}"
    except Exception:
        return f"{k}{sep}{type(v)}"


def add_prefix(evals: EvalsType, prefix: str) -> EvalsType:
    """metric名にprefixを付ける。"""
    return {f"{prefix}{k}": v for k, v in evals.items()}


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
