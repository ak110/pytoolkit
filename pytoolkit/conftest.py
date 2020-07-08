import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))


@pytest.fixture()
def check_dir():
    """目視確認用の結果のディレクトリ。"""
    return pathlib.Path(__file__).resolve().parent.parent / "___check"


@pytest.fixture()
def data_dir():
    """テストデータのディレクトリ。"""
    return pathlib.Path(__file__).resolve().parent / "_test_data"


if True:  # pylint: disable=using-constant-test
    import pytoolkit as tk

    tk.math.set_ndarray_format()
    tk.math.set_numpy_error()
