"""前処理＋モデル＋後処理のパイプライン。"""
import pathlib

import sklearn.pipeline
import numpy as np

import pytoolkit as tk


class Model:
    """パイプラインのモデルのインターフェース。"""

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        raise NotImplementedError()

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        raise NotImplementedError()

    def predict_oof(self, dataset, folds):
        """out-of-foldなpredict結果を返す。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex

        Returns:
            np.ndarray: 予測結果

        """
        pred_list = self.predict(dataset)
        assert len(pred_list) == len(folds)

        oofp_shape = (len(dataset),) + pred_list[0].shape[1:]
        oofp = np.empty(oofp_shape, dtype=pred_list[0].dtype)
        for pred, (_, val_indices) in zip(pred_list, folds):
            oofp[val_indices] = pred[val_indices]

        return oofp

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        raise NotImplementedError()


class Pipeline(Model):
    """前処理＋モデル＋後処理のパイプライン。

    Args:
        model (Model): モデル
        preprocessors (list): 前処理 (sklearnのTransformerの配列)
        postprocessors (list): 後処理 (sklearnのTransformerの配列)

    """

    def __init__(self, model, preprocessors=None, postprocessors=None):
        self.model = model
        self.preprocessors = (
            sklearn.pipeline.make_pipeline(*preprocessors)
            if preprocessors is not None
            else None
        )
        self.postprocessors = (
            sklearn.pipeline.make_pipeline(*postprocessors)
            if postprocessors is not None
            else None
        )

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        if models_dir is not None:
            models_dir = pathlib.Path(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)

        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.fit_transform(
                dataset.data, dataset.labels
            )
        if self.postprocessors is not None:
            dataset.labels = np.squeeze(
                self.postprocessors.fit_transform(
                    np.expand_dims(dataset.labels, axis=-1)
                ),
                axis=-1,
            )
        scores = self.model.cv(dataset, folds, models_dir)

        if models_dir is not None:
            if self.preprocessors is not None:
                tk.utils.dump(self.preprocessors, models_dir / "preprocessors.pkl")
            if self.postprocessors is not None:
                tk.utils.dump(self.postprocessors, models_dir / "postprocessors.pkl")

        return scores

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (PathLike): 保存先ディレクトリ

        Returns:
            Pipeline: self

        """
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessors = tk.utils.load(
            models_dir / "preprocessors.pkl", skip_not_exist=True
        )
        self.postprocessors = tk.utils.load(
            models_dir / "postprocessors.pkl", skip_not_exist=True
        )
        self.model.load(models_dir)
        return self

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.transform(dataset.data)

        pred_list = self.model.predict(dataset)

        if self.postprocessors is not None:
            for i in range(len(pred_list)):
                if pred_list[i].ndim <= 1:
                    pred_list[i] = np.squeeze(
                        self.postprocessors.inverse_transform(
                            np.expand_dims(pred_list[i], axis=-1)
                        ),
                        axis=-1,
                    )
                else:
                    pred_list[i] = self.postprocessors.inverse_transform(pred_list[i])

        return pred_list


class SKLearnModel:
    """scikit-learnのモデル。

    Args:
        estimator (sklearn.base.BaseEstimator): モデル
        weights_arg_name (str): tk.data.Dataset.weightsを使う場合の引数名
                                (pipelineなどで変わるので。例: "transformedtargetregressor__sample_weight")

    """

    def __init__(self, estimator, weights_arg_name="sample_weight"):
        self.estimator = estimator
        self.weights_arg_name = weights_arg_name
        self.estimators_ = None

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        scores = []
        score_weights = []
        self.estimators_ = []
        for train_indices, val_indices in tk.utils.tqdm(folds, desc="cv"):
            train_set = dataset.slice(train_indices)
            val_set = dataset.slice(val_indices)

            kwargs = {}
            if train_set.weights is not None:
                kwargs[self.weights_arg_name] = train_set.weights

            estimator = sklearn.base.clone(self.estimator)
            estimator.fit(train_set.data, train_set.labels, **kwargs)
            self.estimators_.append(estimator)

            kwargs = {}
            if val_set.weights is not None:
                kwargs[self.weights_arg_name] = val_set.weights

            scores.append(estimator.score(val_set.data, val_set.labels, **kwargs))
            score_weights.append(len(val_set))

        if models_dir is not None:
            tk.utils.dump(self.estimators_, models_dir / "estimators.pkl")

        return {"score": np.average(scores, weights=score_weights)}

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        self.estimators_ = tk.utils.load(models_dir / "estimators.pkl")

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        # TODO: predict_proba対応
        return np.array(
            [estimator.predict(dataset.data) for estimator in self.estimators_]
        )


class KerasModel:
    """Kerasのモデル。

    Args:
        create_model_fn (callable): モデルを作成する関数。
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        batch_size (int): バッチサイズ
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists (bool): モデルが存在してもスキップせず再学習するならFalse。
        fit_params (dict): tk.models.fit()のパラメータ

    """

    def __init__(
        self,
        create_model_fn,
        train_preprocessor=None,
        val_preprocessor=None,
        batch_size=32,
        *,
        model_name_format="model.fold{fold}.h5",
        skip_if_exists=True,
        load_model_fn=None,
        use_horovod=False,
        fit_params=None,
    ):
        self.create_model_fn = create_model_fn
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.batch_size = batch_size
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.fit_params = fit_params
        self.load_model_fn = load_model_fn
        self.use_horovod = use_horovod
        self.models_dir_ = None  # TODO: TF2.0でsession廃止してリファクタリング
        self.nfold_ = None  # TODO: TF2.0でsession廃止してリファクタリング

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (None未対応)

        Returns:
            dict: metrics名と値

        """
        tk.training.cv(
            create_model_fn=self.create_model_fn,
            train_dataset=dataset,
            folds=folds,
            train_preprocessor=self.train_preprocessor,
            val_preprocessor=self.val_preprocessor,
            batch_size=self.batch_size,
            models_dir=models_dir,
            model_name_format=self.model_name_format,
            skip_if_exists=self.skip_if_exists,
            **(self.fit_params or {}),
        )
        return {}  # TODO: score

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        self.models_dir_ = models_dir
        # foldを数える (仮)
        for fold in range(999):
            path = models_dir / self.model_name_format.format(fold=fold)
            if path.exists():
                self.nfold_ = fold + 1
            else:
                break

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        return tk.training.predict_cv(
            dataset=dataset,
            nfold=self.nfold_,
            preprocessor=self.val_preprocessor,
            batch_size=self.batch_size,  # TODO: 2倍？
            load_model_fn=self.load_model_fn,
            use_horovod=self.use_horovod,
            models_dir=self.models_dir_,
            oof=False,
        )
