"""JSONの薄いラッパー。"""
import json
import pathlib

import numpy as np


def dump(value, filename, ensure_ascii=True, indent=2, sort_keys=True, separators=(',', ': ')):
    """保存。"""
    with pathlib.Path(filename).open('w', encoding='utf-8') as f:
        json.dump(value, f, default=_default, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def dumps(value, ensure_ascii=True, indent=2, sort_keys=True, separators=(',', ': ')):
    """保存。"""
    return json.dumps(value, default=_default, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def load(filename):
    """読み込み。"""
    with pathlib.Path(filename).open('r', encoding='utf-8') as f:
        return json.load(f)


def loads(s):
    """読み込み。"""
    return json.loads(s)


def _default(o):
    """整形。"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)
