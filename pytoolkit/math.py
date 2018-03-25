"""主に機械学習関連？"""
import typing

import numpy as np


def print_histgram(a, bins=10, range=None, weights=None, ncols=72, name=None, print_fn=print):  # pylint: disable=W0622
    """ヒストグラムを表示する。"""
    for line in format_histgram(a, bins=bins, range=range, weights=weights, ncols=ncols, name=name):
        print_fn(line)


def format_histgram(a, bins=10, range=None, weights=None, ncols=72, name=None) -> typing.Sequence[str]:  # pylint: disable=W0622
    """ヒストグラムをテキストで返す。"""
    hist, bin_edges = np.histogram(a, bins=bins, range=range, weights=weights)
    hist = np.asarray(hist)
    edges_text = format_values(bin_edges)

    std_hist = hist / hist.sum()
    norm_hist = hist / hist.max()
    has_100 = (std_hist >= 0.995).any()
    max_bar_size = ncols - (len(edges_text[0]) + 2 + len(edges_text[0]) + 2 + (5 if has_100 else 4) + 4)

    lines = []
    if name is not None:
        n = (ncols - len(name)) // 2 - 1
        lines.append(f'{"-" * n} {name} {"-" * n}')
    for i, (sh, nh) in enumerate(zip(std_hist, norm_hist)):
        percent = format(sh * 100, f'{5 if has_100 else 4}.1f')
        bar_size = int(round(max_bar_size * nh))
        lines.append(f'{edges_text[i]}～{edges_text[i+1]} ({percent}%)  {"#" * bar_size}')
    return lines


def format_values(values: typing.Union[list, np.ndarray]):
    """`values`をいい感じに固定長の文字列にして返す。"""
    values = np.asarray(values)
    assert len(values.shape) == 1

    has_minus = (values < 0).any()
    abs_values = np.abs(values)

    if (abs_values < 0.1).all() or (abs_values >= 10).any():
        # 指数表示
        fmt = f'{9 if has_minus else 8}.2e'
    else:
        # 少数表示
        fmt = f'{6 if has_minus else 5}.3f'

    formatted = [format(x, fmt) for x in values]
    return formatted
