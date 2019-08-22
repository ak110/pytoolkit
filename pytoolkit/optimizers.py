"""Kerasのoptimizer関連。"""
# pylint: disable=cell-var-from-loop,attribute-defined-outside-init

from . import K, keras


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = []
    return {c.__name__: c for c in classes}
