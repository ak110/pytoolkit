import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).parent.parent))


@pytest.fixture()
def check_dir():
    """目視確認用の結果のディレクトリ。"""
    return pathlib.Path(__file__).resolve().parent.parent / "___check"


@pytest.fixture()
def data_dir():
    """テストデータのディレクトリ。"""
    return pathlib.Path(__file__).resolve().parent / "_test_data"
