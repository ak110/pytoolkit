"""Tablur Data用NN。

Examples:
    ::

        import pytoolkit.tf
        import pytoolkit.tablurs

        train_data_path = "path/to/train.csv"
        test_data_path = "path/to/test.csv"
        input_data_path = "path/to/input.csv"

        # ログの初期化
        pytoolkit.logs.init()

        # データの読み込み
        train_data, train_labels = pytoolkit.tablurs.load_labeled_data(
            train_data_path, "label_col_name"
        )
        test_data, test_labels = pytoolkit.tablurs.load_labeled_data(
            test_data_path, "label_col_name"
        )

        # 学習
        model = pytoolkit.tf.tdnn.train(train_data, train_labels, groups=None)

        # 保存・読み込み
        model.save(model_dir)
        model = pytoolkit.tf.tdnn.load(model_dir)

        # 評価
        score = model.evaluate(test_data, test_labels)
        assert 0.0 <= score <= 1.0

        # 推論
        input_data = pytoolkit.tablurs.load_unlabeled_data(input_data_path)
        results = model.infer(input_data)
        assert isinstance(results, np.ndarray)

"""
import json
import logging
import os
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import sklearn.metrics
import sklearn.model_selection
import tensorflow as tf
import tensorflow_addons as tfa

import pytoolkit.tf

logger = logging.getLogger(__name__)


class Model:
    """テーブルデータのモデル。"""

    def __init__(
        self,
        train_model: tf.keras.models.Model,
        infer_model: tf.keras.models.Model,
        metadata: dict[str, typing.Any],
    ):
        self.train_model = train_model
        self.infer_model = infer_model
        self.metadata = metadata

    def save(self, model_dir: str | os.PathLike[str]) -> None:
        """保存。

        Args:
            model_dir: 保存先ディレクトリ

        """
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.infer_model.save_weights(str(model_dir / "model.keras"))

        (model_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, model_dir: str | os.PathLike[str]) -> "Model":
        """モデルの読み込み

        Args:
            model_dir: 保存先ディレクトリ

        Returns:
            モデル

        """
        model_dir = pathlib.Path(model_dir)
        metadata: dict[str, typing.Any] = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        train_model, infer_model = _create_model(metadata)
        infer_model.load_weights(str(model_dir / "model.keras"))
        return cls(train_model, infer_model, metadata)

    def evaluate(
        self, data: pd.DataFrame | pl.DataFrame, labels: npt.ArrayLike
    ) -> dict[str, float]:
        """推論。

        Args:
            data: 入力データ
            labels: ラベル

        Returns:
            スコア

        """
        pred = self.infer(data)
        if self.metadata["task"] == "binary":
            labels = _class_to_index(labels, self.metadata["class_names"])
            return {
                "acc": float(sklearn.metrics.accuracy_score(labels, np.round(pred))),
                "auc": float(sklearn.metrics.roc_auc_score(labels, pred)),
            }
        elif self.metadata["task"] == "multiclass":
            labels = _class_to_index(labels, self.metadata["class_names"])
            return {
                "acc": float(
                    sklearn.metrics.accuracy_score(labels, pred.argmax(axis=-1))
                ),
                "auc": float(
                    sklearn.metrics.roc_auc_score(labels, pred, multi_class="ovo")
                ),
            }
        else:
            assert self.metadata["task"] == "regression"
            return {
                "mae": float(sklearn.metrics.mean_absolute_error(labels, pred)),
                "rmse": float(
                    sklearn.metrics.mean_squared_error(labels, pred, squared=False)
                ),
                "r2": float(sklearn.metrics.r2_score(labels, pred)),
            }

    def infer(
        self, data: pd.DataFrame | pl.DataFrame, verbose: bool = True
    ) -> npt.NDArray[np.float32]:
        """推論。

        Args:
            data: 入力データ
            verbose: 進捗表示の有無

        Returns:
            推論結果

        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        features = _preprocess(data, self.metadata)
        global_batch_size = (
            self.metadata["batch_size"]
            * tf.distribute.get_strategy().num_replicas_in_sync
        )
        ds = _create_dataset(features, global_batch_size)
        return self.infer_model.predict(ds, verbose=1 if verbose else 0)

    def infer_oof(
        self,
        data: pd.DataFrame | pl.DataFrame,
        folds: typing.Sequence[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]],
        verbose: bool = True,
    ) -> npt.NDArray[np.float32]:
        """out-of-fold推論。

        Args:
            data: 入力データ
            folds: 分割方法
            verbose: 進捗表示の有無

        Returns:
            推論結果

        """
        assert len(folds) == self.metadata["num_folds"]
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        features = _preprocess(data, self.metadata)
        global_batch_size = (
            self.metadata["batch_size"]
            * tf.distribute.get_strategy().num_replicas_in_sync
        )
        ds = _create_dataset(features, global_batch_size, folds=folds)
        return self.train_model.predict(ds, verbose=1 if verbose else 0)

    def get_feature_names(self) -> list[str]:
        """列名を返す。

        Returns:
            列名の配列

        """
        return self.metadata["feature_names"]


def load_labeled_data(
    data_path: str | os.PathLike[str], label_col_name: str
) -> tuple[pl.DataFrame, npt.NDArray]:
    """ラベルありデータの読み込み

    Args:
        data_path: データのパス(CSV, Excelなど)
        label_col_name: ラベルの列名

    Returns:
        データフレーム

    """
    data = load_unlabeled_data(data_path)
    labels = data.drop_in_place(label_col_name).to_numpy()
    return data, labels


def load_unlabeled_data(data_path: str | os.PathLike[str]) -> pl.DataFrame:
    """ラベルなしデータの読み込み

    Args:
        data_path: データのパス(CSV, Excelなど)

    Returns:
        データフレーム

    """
    data_path = pathlib.Path(data_path)
    data: pl.DataFrame
    if data_path.suffix.lower() == ".csv":
        data = pl.read_csv(data_path)
    elif data_path.suffix.lower() == ".tsv":
        data = pl.read_csv(data_path, sep="\t")
    elif data_path.suffix.lower() == ".arrow":
        data = pl.read_ipc(data_path)
    elif data_path.suffix.lower() == ".parquet":
        data = pl.read_parquet(data_path)
    elif data_path.suffix.lower() in (".xls", ".xlsx", ".xlsm"):
        data = pl.read_excel(data_path)  # type: ignore
    else:
        raise ValueError(f"Unknown suffix: {data_path}")
    return data


def train(
    data: pd.DataFrame | pl.DataFrame,
    labels: npt.ArrayLike,
    folds: typing.Sequence[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]
    | None = None,
    categorical_feature: list[str] | None = None,
    epochs: int = 100,
) -> Model:
    """学習

    Args:
        data: 入力データ
        labels: ラベル
        weights: 入力データの重み
        groups: 入力データのグループ
        folds: CVの分割情報
        categorical_feature: カテゴリ列

    Returns:
        モデル

    """
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    labels = np.asarray(labels)
    assert len(data) == len(labels), f"{len(data)=} {len(labels)=}"

    if folds is None:
        kf = sklearn.model_selection.StratifiedKFold()
        folds = list(kf.split(data, labels))  # type: ignore

    metadata: dict[str, typing.Any] = {
        "num_folds": len(folds),
        "feature_names": data.columns.to_list(),
        "batch_size": 128,
    }

    early_stopping = False
    if isinstance(labels[0], str):
        # 分類の場合
        class_names: list[str] = np.sort(np.unique(labels)).tolist()
        metadata["class_names"] = class_names
        labels = _class_to_index(labels, class_names)
        assert len(class_names) >= 2
        if len(class_names) == 2:
            # 2クラス分類
            metadata["task"] = "binary"
            metadata["num_outputs"] = 1
            loss = "binary_crossentropy"
            metrics = ["acc"]
        else:
            # 多クラス分類
            metadata["task"] = "multiclass"
            metadata["num_outputs"] = len(class_names)
            loss = "sparse_categorical_crossentropy"
            metrics = ["acc"]
    else:
        # 回帰の場合
        assert labels.dtype.type is np.float32
        metadata["task"] = "regression"
        metadata["num_outputs"] = 1
        if (0.0 <= labels.min() <= 0.25) and (0.75 <= labels.max() <= 1.0):
            # 0～1に特化した実装
            metadata["regression_sigmoid"] = True
            loss = "binary_crossentropy"
            logger.info("tdnn: regression sigmoid")
        else:
            # TODO: とりあえず単純に標準化してlog_cosh
            metadata["regression_sigmoid"] = False
            metadata["regression_shift"] = labels.mean().tolist()
            metadata["regression_scale"] = labels.std(ddof=1).tolist() + 1e-1
            labels = (labels - metadata["regression_shift"]) / metadata[
                "regression_scale"
            ]
            logger.info(
                f'tdnn: regression shift={metadata["regression_shift"]:.1f}'
                f' scale={metadata["regression_scale"]:.1f}'
            )
            loss = "log_cosh"
        metrics = ["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
        early_stopping = True

    # カテゴリ列のエンコード
    metadata["encode_categoricals"] = {}
    # TODO: 本当はラベルエンコード以外もできるようにしたい
    if categorical_feature is None:
        categorical_feature = []
    else:
        for c in categorical_feature:
            values = np.sort(data[c].dropna().unique()).tolist()
            logger.info(f"tdnn: encode_categoricals({c}) => {values}")
            metadata["encode_categoricals"][c] = values

    # カテゴリ列以外の前処理
    metadata["feature_scales"] = {}
    metadata["isnull_features"] = []
    for c in data.columns.to_list():
        values = data[c].to_numpy()
        # とりあえずただの標準化 (TODO: 改善)
        na_values = values[np.isfinite(values)]
        if len(na_values) <= 1:
            moments = [0.0, 1.0]
        else:
            moments = [
                np.mean(na_values).tolist(),
                np.std(na_values, ddof=1).tolist() + 1e-1,
            ]
            logger.info(
                f"tdnn: feature scale({c}) =>"
                f" mean={moments[0]:.1f} std={moments[1]:.1f}"
            )
        metadata["feature_scales"][c] = moments

        # isnull列
        na_ratio = len(na_values) / len(values)
        if 0.05 < na_ratio < 0.95:
            metadata["isnull_features"].append(c)
            logger.info(f"tdnn: isnull feature({c}) => {na_ratio=:.0%}")

    features = _preprocess(data, metadata)
    metadata["num_features"] = features.shape[-1]

    train_model, infer_model = _create_model(metadata)

    global_batch_size = (
        metadata["batch_size"] * tf.distribute.get_strategy().num_replicas_in_sync
    )
    train_steps = -(-len(features) // global_batch_size)
    train_model.summary(print_fn=logger.info)
    train_model.compile(
        tfa.optimizers.RectifiedAdam(
            1e-4 * global_batch_size**0.5,
            min_lr=1e-6 * global_batch_size**0.5,
            warmup_proportion=0.1,
            total_steps=train_steps * epochs,
            weight_decay=1e-5,
        ),
        loss,
        metrics,
    )
    train_model.fit(
        _create_dataset(features, global_batch_size, labels, folds, shuffle=True),
        steps_per_epoch=train_steps,
        validation_data=_create_dataset(features, global_batch_size, labels, folds),
        validation_freq=1
        if early_stopping
        else [1] + list(range(epochs, 1, -int(epochs**0.5))),
        epochs=epochs,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=max(
                    10, int(epochs * train_model.optimizer.warmup_proportion) + 1
                ),
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1,
                # start_from_epoch=int(epochs * 0.1),
            )
        ]
        if early_stopping
        else [],
    )

    model = Model(train_model, infer_model, metadata)
    return model


def load(model_dir: str | os.PathLike[str]) -> Model:
    """学習済みモデルの読み込み

    Args:
        model_dir: 保存先ディレクトリ

    Returns:
        モデル

    """
    return Model.load(model_dir)


def _class_to_index(
    labels: npt.ArrayLike, class_names: list[str]
) -> npt.NDArray[np.int32]:
    return np.vectorize(class_names.index)(labels)


def _create_model(
    metadata: dict[str, typing.Any]
) -> tuple[tf.keras.models.Model, tf.keras.models.Model]:
    input_features = tf.keras.Input((metadata["num_features"],), name="features")
    input_fold = tf.keras.Input((), dtype="int32", name="fold")

    fold_models = [
        _create_fold_model(metadata, fold_index)
        for fold_index in range(metadata["num_folds"])
    ]
    x = pytoolkit.tf.layers.CVMerge()(
        [
            m(pytoolkit.tf.layers.CVPick(fold_index)([input_features, input_fold]))
            for fold_index, m in enumerate(fold_models)
        ]
        + [input_fold]
    )
    if metadata["num_outputs"] <= 1:
        x = tf.squeeze(x, axis=-1)
    train_model = tf.keras.models.Model([input_features, input_fold], x)

    x = tf.keras.layers.average([m(input_features) for m in fold_models])
    if metadata["num_outputs"] <= 1:
        x = tf.squeeze(x, axis=-1)
    infer_model = tf.keras.models.Model(input_features, x)

    return train_model, infer_model


def _create_fold_model(
    metadata: dict[str, typing.Any], fold_index: int
) -> tf.keras.models.Model:
    inputs = x = tf.keras.Input((metadata["num_features"],))

    # https://www.kaggle.com/code/sishihara/1dcnn-for-tabular-from-moa-2nd-place
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Reshape((256, 16))(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=5, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)

    x = tf.keras.layers.Dense(metadata["num_outputs"])(x)

    if metadata["task"] == "binary":
        x = tf.keras.layers.Activation("sigmoid")(x)
    elif metadata["task"] == "multiclass":
        x = tf.keras.layers.Activation("softmax")(x)
    elif metadata["regression_sigmoid"]:
        assert metadata["task"] == "regression"
        x = tf.keras.layers.Activation("sigmoid")(x)
    else:
        assert metadata["task"] == "regression"
        x = x * metadata["regression_scale"] + metadata["regression_shift"]

    return tf.keras.models.Model(inputs, x, name=f"fold_model_{fold_index + 1}")


def _create_dataset(
    features: np.ndarray, global_batch_size, labels=None, folds=None, shuffle=False
) -> tf.data.Dataset:
    if folds is None:
        inputs = {"features": features}
    else:
        fold = np.full((len(features),), -1, dtype=np.int32)
        for i, (_, val_indices) in enumerate(folds):
            fold[val_indices] = i
        inputs = {"features": features, "fold": fold}
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(inputs)
    else:
        assert len(features) == len(labels)
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features)).repeat()
    ds = ds.batch(global_batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    ds = ds.with_options(options)
    return ds


def _preprocess(data: pd.DataFrame, metadata: dict[str, typing.Any]) -> np.ndarray:
    """前処理。"""
    data = data.copy()
    features = []

    for c, values in metadata["encode_categoricals"].items():
        features.append(
            data[c]
            .map(values.index, na_action="ignore")
            .fillna(-1)
            .to_numpy(dtype=np.float32)
        )

    for c, (shift, scale) in metadata["feature_scales"].items():
        features.append(
            ((data[c] - shift) / scale).fillna(0.0).to_numpy(dtype=np.float32)
        )

    for c in metadata["isnull_features"]:
        features.append(data[c].isnull().to_numpy(dtype=np.float32))

    return np.stack(features, axis=-1)
