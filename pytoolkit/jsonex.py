"""JSONの薄いラッパー。"""
import json
import pathlib

import numpy as np


def dump(value, filename, ensure_ascii=False, indent=2, sort_keys=True, separators=(',', ': ')):
    """保存。"""
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('w', encoding='utf-8') as f:
        json.dump(value, f, default=_default, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def dumps(value, ensure_ascii=False, indent=2, sort_keys=True, separators=(',', ': ')):
    """保存。"""
    return json.dumps(value, default=_default, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def load(filename):
    """読み込み。"""
    filename = pathlib.Path(filename)
    with filename.open('r', encoding='utf-8') as f:
        return json.load(f)


def loads(s):
    """読み込み。"""
    return json.loads(s)


def _default(o):
    """整形。"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)
