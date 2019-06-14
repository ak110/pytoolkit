"""LightGBMなどなど関連。"""


class ModelExtractionCallback:
    """lightgbm.cv() から学習済みモデルを取り出すためのコールバックに使うクラス

    NOTE: 非公開クラス '_CVBooster' に依存しているため将来的に動かなく恐れがある

    References:
        - <https://blog.amedama.jp/entry/lightgbm-cv-model>

    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster の参照を保持する
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # コールバックが呼ばれていないときは例外にする
            raise RuntimeError("callback has not called yet")

    @property
    def boosters_proxy(self):
        """Booster へのプロキシオブジェクトを返す。"""
        self._assert_called_cb()
        return self._model

    @property
    def raw_boosters(self):
        """Booster のリストを返す。"""
        self._assert_called_cb()
        return self._model.boosters

    @property
    def best_iteration(self):
        """Early stop したときの boosting round を返す。"""
        self._assert_called_cb()
        return self._model.best_iteration
