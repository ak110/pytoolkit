import pathlib

import pytest

import pytoolkit as tk


@pytest.fixture(scope='session', autouse=True)
def global_setup():
    """pytestセッション全体での初期化。"""
    import matplotlib
    matplotlib.use('Agg')


@pytest.fixture()
def dl_session():
    """関数ごとにsessionするfixture。"""
    with tk.dl.session():
        yield


@pytest.fixture()
def check_dir():
    """目視確認用の結果のディレクトリ。"""
    return pathlib.Path(__file__).resolve().parent.parent / '___check'


@pytest.fixture()
def data_dir():
    """テストデータのディレクトリ。"""
    return pathlib.Path(__file__).resolve().parent / 'data'