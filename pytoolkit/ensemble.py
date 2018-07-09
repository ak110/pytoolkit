"""スタッキングとか関連。"""
import abc
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

from . import dl, log, ml, utils


class WeakModel(metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def fit(self, X, y):
        """学習。"""
        utils.noqa(X)
        utils.noqa(y)

    @abc.abstractmethod
    def predict(self, X):
        """予測。"""
        utils.noqa(X)

    def split(self, X, y, cv_index):
        """クロスバリデーションの1fold分のデータを返す。"""
        assert cv_index in range(self.cv_count)
        train_indices, val_indices = ml.cv_indices(
            X, y, self.cv_count, cv_index, self.split_seed, stratify=self.stratify)
        logger = log.get(__name__)
        logger.info(f'Training data count:   {len(train_indices):7d}')
        logger.info(f'Validation data count: {len(val_indices):7d}')
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        return (X_train, y_train), (X_val, y_val)

    def has_prediction(self, cv_index):
        """予測結果が存在するか否かを返す。"""
        return (self.model_dir / f'proba_val.fold{cv_index}.pkl').is_file()

    def save_prediction(self, proba_val, cv_index):
        """予測結果の保存。"""
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


class WeakModelManager(metaclass=abc.ABCMeta):
    """WeakModelたちを管理するクラス。

    - models_dir: 各モデルの処理結果などの保存先

    """

    def __init__(self, models_dir):
        self.models_dir = pathlib.Path(models_dir)
        self.models = {}

    def add(self, model: WeakModel):
        """WeakModelの追加。"""
        assert model.name not in self.models, f'WeakModel duplicate name error: {model.name}'
        self.models[model.name] = model
        model.set_manager(self)

    def get(self, name: str) -> WeakModel:
        """WeakModelの取得。"""
        assert name in self.models, f'WeakModel name error: {name}'
        return self.models[name]

    @property
    def model_names(self) -> tuple:
        """モデルの名前のタプルを返す。"""
        return tuple(sorted(self.models.keys()))

    def load_oopf(self, X, y, names):
        """スタッキング用に各モデルのout-of-folds predictionを読み込んでくっつけて返す。"""
        oopf_list = [self.get(name).load_oopf(X, y) for name in names]
        return np.concatenate(oopf_list, axis=-1)

    def predict(self, X, names):
        """スタッキング用に各モデルのpredictを読み込んでくっつけて返す。"""
        mf_list_list = [self.get(name).predict(X) for name in names]
        mf_list = [np.concatenate(mf_list, axis=-1) for mf_list in zip(*mf_list_list)]
        return mf_list

    @abc.abstractmethod
    def evaluate(self, y_true, proba_pred):
        """予測結果の評価。"""
        utils.noqa(y_true)
        utils.noqa(proba_pred)


class KerasWeakModel(WeakModel):
    """KerasなWeakModel。"""

    def fit(self, X, y):
        for cv_index in range(self.cv_count):
            if self.has_prediction(cv_index):
                pass
            else:
                (X_train, y_train), (X_val, y_val) = self.split(X, y, cv_index)
                # 学習
                with self.session(train=True):
                    self.fit_fold(X_train, y_train, X_val, y_val, cv_index)
                # 検証
                if dl.hvd.is_master():
                    with self.session(train=False):
                        proba_val = self.predict_fold(X_val, cv_index)
                    self.manager.evaluate(y_val, proba_val)
                    self.save_prediction(proba_val, cv_index)
                dl.hvd.barrier()

    def predict(self, X):
        """予測。"""
        return [self.predict_fold(X, cv_index=cv_index) for cv_index in range(self.cv_count)]

    def session(self, train):
        """オプション変えるとき用。"""
        utils.noqa(self)
        return dl.session(use_horovod=train)

    @abc.abstractmethod
    def fit_fold(self, X_train, y_train, X_val, y_val, cv_index):
        """1-fold分の学習をしてX_valに対する予測結果を返す。"""
        utils.noqa(X_train)
        utils.noqa(y_train)
        utils.noqa(X_val)
        utils.noqa(y_val)
        utils.noqa(cv_index)

    @abc.abstractmethod
    def predict_fold(self, X, cv_index):
        """1-fold分の予測。テストデータの場合はcv_index=-1。"""
        utils.noqa(X)
        utils.noqa(cv_index)
        return None
