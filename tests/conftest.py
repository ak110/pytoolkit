import pathlib
import sys

import pytest


sys.path.append(str(pathlib.Path(__file__).parent.parent))

if True:
    import pytoolkit as tk

    @pytest.fixture(scope="session", autouse=True)
    def global_setup():
        """pytestセッション全体での初期化。"""
        import matplotlib

        matplotlib.use("Agg")

    @pytest.fixture()
    def session():
        """関数ごとにtf.sessionするfixture。"""
        with tk.dl.session() as s:
            yield s.session

    @pytest.fixture()
    def check_dir():
        """目視確認用の結果のディレクトリ。"""
        return pathlib.Path(__file__).resolve().parent.parent / "___check"

    @pytest.fixture()
    def data_dir():
        """テストデータのディレクトリ。"""
        return pathlib.Path(__file__).resolve().parent / "data"
