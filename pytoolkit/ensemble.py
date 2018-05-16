"""スタッキングとか関連。"""
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

from . import dl, log, ml, utils


class WeakModel(object):
    """CVしたりout-of-fold predictionを作ったりするクラス。

    継承してfit()とかpredict()とかを実装する。

    """

    def __init__(self, name: str, cv_count: int, split_seed: int, stratify: bool):
        self.name = name
        self.cv_count = cv_count
        self.split_seed = split_seed
        self.stratify = stratify
        self.manager = None

    def set_manager(self, manager):
        """WeakModelManagerの設定。(WeakModelManager.add()から呼ぶので手動で呼ぶ必要は無い。)"""
        self.manager = manager

    @property
    def model_dir(self) -> pathlib.Path:
        """処理結果などの保存先を返す。"""
        model_dir = self.manager.models_dir / self.name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def fit(self, X, y):
        """クロスバリデーション。"""
        for cv_index in range(self.cv_count):
            self.fit_fold(X, y, cv_index)

    def fit_fold(self, X, y, cv_index):
        """クロスバリデーションの1fold分の処理。"""
        assert cv_index in range(self.cv_count)
        # データ分割
        train_indices, val_indices = ml.cv_indices(
            X, y, self.cv_count, cv_index, self.split_seed, stratify=self.stratify)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        logger = log.get(__name__)
        logger.info(f'training data count = {len(train_indices)}, validation data count = {len(val_indices)}')
        # 学習＆予測
        proba_val = self.fit_impl(X_train, y_train, X_val, y_val, cv_index)
        # 予測結果の保存
        if dl.hvd.is_master():
            joblib.dump(proba_val, self.model_dir / f'proba_val.fold{cv_index}.pkl')
        dl.hvd.barrier()

    def load_oopf(self, X, y):
        """学習時に作成したout-of-folds predictionを読み込んで返す。"""
        proba_val_list = [joblib.load(self.model_dir / f'proba_val.fold{cv_index}.pkl')
                          for cv_index in range(self.cv_count)]

        proba_val = np.empty((len(y),) + proba_val_list[0].shape[1:])
        for cv_index in range(self.cv_count):
            _, val_indices = ml.cv_indices(X, y, self.cv_count, cv_index, self.split_seed, stratify=self.stratify)
            proba_val[val_indices] = proba_val_list[cv_index]

        return proba_val

    def fit_impl(self, X_train, y_train, X_val, y_val, cv_index):
        """学習の実装。X_valに対する予測結果を返す。"""
        utils.noqa(X_train)
        utils.noqa(y_train)
        utils.noqa(X_val)
        utils.noqa(y_val)
        utils.noqa(cv_index)
        assert False

    def predict(self, X):
        """予測の実装。"""
        utils.noqa(X)
        assert False


class WeakModelManager(object):
    """WeakModelたちを管理するクラス。

    - models_dir: 各モデルの処理結果などの保存先

    """

    def __init__(self, models_dir):
        self.models_dir = pathlib.Path(models_dir)
        self.models = {}

    def add(self, model: WeakModel):
        """WeakModelの追加。"""
        self.models[model.name] = model
        model.set_manager(self)

    def get(self, name: str) -> WeakModel:
        """WeakModelの取得。"""
