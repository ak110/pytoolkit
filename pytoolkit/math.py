"""主にnumpy関連？"""
import typing

import numpy as np
import scipy.stats


def between(x, a, b):
    """`a <= x <= b`を返す。"""
    assert np.all(a <= b)
    return np.logical_and(a <= x, x <= b)


def in_range(x, a, b):
    """`a <= x < b`を返す。"""
    assert np.all(a < b)
    return np.logical_and(a <= x, x < b)


def print_histgram(a, bins=10, range=None, weights=None, with_stats=True, ncols=72, name=None, print_fn=print):  # pylint: disable=W0622
    """ヒストグラムを表示する。"""
    for line in format_histgram(a, bins=bins, range=range, weights=weights, with_stats=with_stats, ncols=ncols, name=name):
        print_fn(line)


def format_histgram(a, bins=10, range=None, weights=None, with_stats=True, ncols=72, name=None) -> typing.Sequence[str]:  # pylint: disable=W0622
    """ヒストグラムをテキストで返す。"""
    hist, bin_edges = np.histogram(a, bins=bins, range=range, weights=weights)
    hist = np.asarray(hist)
    edges_text = format_values(bin_edges)

    std_hist = hist / (hist.sum() + 1e-7)
    norm_hist = hist / (hist.max() + 1e-7)
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
    if with_stats:
        lines.append(f'mean: {np.mean(a)}')
        lines.append(f'std:  {np.std(a)}')
    return lines


def format_values(values: typing.Union[list, np.ndarray], padding_sign=True):
    """`values`をいい感じに固定長の文字列にして返す。"""
    values = np.asarray(values)
    assert len(values.shape) == 1

    has_minus = (values < 0).any()
    minus_col = 1 if has_minus and padding_sign else 0
    abs_values = np.abs(values)

    if issubclass(values.dtype.type, np.integer) and abs_values.max() <= 99999999:
        n = int(np.ceil(np.log10(abs_values.max())))
        fmt = f'{n + minus_col}d'
    elif (abs_values < 0.1).all() or (abs_values >= 10).any():
        # 指数表示
        fmt = f'{8 + minus_col}.2e'
    else:
        # 少数表示
        fmt = f'{5 + minus_col}.3f'

    formatted = [format(x, fmt) for x in values]
    return formatted


def binorm_percent(p, positives, total):
    """二項分布のp%信頼区間をパーセントの整数値で返す。(「x±y%」形式)

    色々適当に丸めてるので注意。

    - p: 有意水準(0.99など)
    - positives: 正解数など
    - total: 全件数
    """
    interval = binorm_interval(p, positives, total)
    lower = interval[0] * 100
    upper = interval[1] * 100
    center = 100 * positives / total
    width = max(upper - center, center - lower)
    return int(np.floor(np.nan_to_num(center))), int(np.ceil(np.nan_to_num(width)))


def binorm_interval(p, positives, total):
    """二項分布のp%信頼区間を返す。

    - p: 有意水準(0.99など)
    - positives: 正解数など
    - total: 全件数
    """
    assert 0 < p < 1
    assert 0 <= positives <= total
    interval = scipy.stats.binom.interval(p, total + 1, positives / (total + 1))
    interval = np.array(interval) / (total + 1)
    return interval
