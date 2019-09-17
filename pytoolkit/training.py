"""Kerasでの学習周りの便利関数など。"""
from __future__ import annotations

import pathlib
import contextlib
import typing

import numpy as np

import pytoolkit as tk

from . import keras


def check(model: keras.models.Model, plot_path=None):
    """モデルの動作確認など。

    Args:
        model (keras.models.Model): モデル
        plot_path (PathLike object): モデルのplotの保存先パス

    """
    with tk.log.trace_scope("check"):
        # summary表示
        tk.models.summary(model)
        # グラフを出力
        if plot_path is not None:
            try:
                tk.models.plot(model, plot_path)
            except ValueError:
                pass  # "Cannot embed the 'svg' image format" (tf >= 1.14)


def cv(
    create_model_fn: typing.Callable[[], tk.keras.models.Model],
    train_set: tk.data.Dataset,
    folds: tk.validation.FoldsType,
    train_preprocessor: tk.data.Preprocessor,
    val_preprocessor: tk.data.Preprocessor,
    batch_size: int = 32,
    *,
    models_dir,
    model_name_format: str = "model.fold{fold}.h5",
    skip_if_exists: bool = True,
    **kwargs,
):
    """CV。

    モデルが存在すればスキップしつつfoldsに従ってcross validationする。

    1/5 foldだけ実行したいとかがあればfoldsの要素を先頭1個にすればOK。

    Args:
        create_model_fn: モデルを作成する関数。
        train_set: 訓練データ
        train_preprocessor: 訓練データの前処理
        val_preprocessor: 検証データの前処理
        folds: train/valのindexの配列のtupleの配列。(sklearn.model_selection.KFold().split()の結果など)
        batch_size: バッチサイズ
        models_dir (PathLike object): モデルの保存先パス (必須)
        model_name_format: モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists : モデルが存在してもスキップせず再学習するならFalse。
        kwargs: tk.models.fit()のパラメータ

    """
    with tk.log.trace_scope("cv"):
        for fold, (train_indices, val_indices) in enumerate(folds):
            # モデルが存在すればスキップ
            model_path = models_dir / model_name_format.format(fold=fold)
            if skip_if_exists and model_path.exists():
                continue

            with tk.log.trace_scope(f"fold#{fold}"):
                tr = train_set.slice(train_indices)
                vl = train_set.slice(val_indices)
                with tk.dl.session(use_horovod=True):
                    train(
                        model=create_model_fn(),
                        train_set=tr,
                        val_set=vl,
                        train_preprocessor=train_preprocessor,
                        val_preprocessor=val_preprocessor,
                        batch_size=batch_size,
                        model_path=model_path,
                        **kwargs,
                    )


def predict_cv(
    dataset: tk.data.Dataset,
    preprocessor: tk.data.Preprocessor,
    batch_size: int = 32,
    load_model_fn: typing.Callable[[pathlib.Path], tk.keras.models.Model] = None,
    *,
    nfold: int = None,
    models: typing.Sequence[keras.models.Model] = None,
    models_dir=None,
    model_name_format: str = "model.fold{fold}.h5",
    oof: bool = False,
    folds: tk.validation.FoldsType = None,
    use_horovod: bool = False,
    **kwargs,
):
    """CVで作ったモデルで予測。

    Args:
        dataset: データ
        preprocessor: 前処理
        batch_size: バッチサイズ
        load_model_fn: モデルを読み込む関数
        nfold: CVのfold数 (models, folds未指定時必須)
        models: 各foldのモデル
        models_dir (PathLike object): モデルの保存先パス (models未指定時必須)
        model_name_format: モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        oof: out-of-fold predictionならTrueにする。folds必須。
        folds: oof時のみ指定する。train/valのindexの配列のtupleの配列。(sklearn.model_selection.KFold().split()の結果など)
        use_horovod: tk.models.predictの引数
        kwargs: tk.models.predictの引数

    Returns:
        list or ndarray: oof=Falseの場合、予測結果のnfold個の配列。oof=Trueの場合、予測結果のndarray。

    """
    with tk.log.trace_scope("predict_cv"):
        if models is not None:
            assert nfold is None or nfold == len(models)
            assert load_model_fn is None
            assert models_dir is None
            nfold = len(models)
        if oof:
            assert folds is not None
            nfold = len(folds)
        else:
            assert folds is None
            assert nfold is not None
            folds = [(np.arange(0), np.arange(len(dataset))) for _ in range(nfold)]
        load_model_fn = load_model_fn or tk.models.load

        with tk.dl.session(
            use_horovod=use_horovod
        ) if models is None else contextlib.nullcontext():
            if models is None:
                models = [
                    load_model_fn(models_dir / model_name_format.format(fold=fold))
                    for fold in tk.utils.trange(nfold, desc="load models")
                ]

            pred_list: list = []
            val_indices_list = []
            for fold, (model, (_, val_indices)) in enumerate(zip(models, folds)):
                pred = tk.models.predict(
                    model,
                    dataset.slice(val_indices),
                    preprocessor,
                    batch_size,
                    desc=f"{'oofp' if oof else 'predict'}({fold + 1}/{nfold})",
                    use_horovod=use_horovod,
                    **kwargs,
                )
                pred_list.append(pred)
                val_indices_list.append(val_indices)

            if oof:
                # TODO: multi output対応
                output_shape = (len(dataset),) + pred_list[0].shape[1:]
                oofp = np.empty(output_shape, dtype=pred_list[0].dtype)
                for pred, val_indices in zip(pred_list, val_indices_list):
                    oofp[val_indices] = pred
                return oofp
            else:
                return pred_list


def train(
    model: keras.models.Model,
    train_set: tk.data.Dataset,
    train_preprocessor: tk.data.Preprocessor,
    val_set: tk.data.Dataset = None,
    val_preprocessor: tk.data.Preprocessor = None,
    batch_size: int = 32,
    *,
    model_path,
    **kwargs,
) -> typing.Optional[dict]:
    """学習。

    Args:
        model: モデル
        train_set: 訓練データ
        train_preprocessor: 訓練データの前処理
        val_set: 検証データ
        val_preprocessor: 検証データの前処理
        batch_size: バッチサイズ
        model_path (PathLike object): モデルの保存先パス (必須)
        kwargs: tk.models.fit()のパラメータ

    Returns:
        val_setがNoneでなければevaluate結果 (metricsの文字列と値のdict)

    """
    with tk.log.trace_scope("train"):
        assert model_path is not None
        assert (val_set is None) == (val_preprocessor is None)
        # 学習
        tk.log.get(__name__).info(
            f"train: {len(train_set)} samples, val: {len(val_set) if val_set is not None else 0} samples, batch_size: {batch_size}x{tk.hvd.size()}"
        )
        tk.hvd.barrier()
        tk.models.fit(
            model,
            train_set,
            train_preprocessor=train_preprocessor,
            val_set=val_set,
            val_preprocessor=val_preprocessor,
            batch_size=batch_size,
            **kwargs,
        )
        try:
            # 評価
            evaluate(
                model,
                train_set,
                preprocessor=train_preprocessor,
                batch_size=batch_size,
                prefix="",
                use_horovod=True,
            )
            evals = (
                evaluate(
                    model,
                    val_set,
                    preprocessor=val_preprocessor,
                    batch_size=batch_size,
                    prefix="val_",
                    use_horovod=True,
                )
                if val_set is not None and val_preprocessor is not None
                else None
            )
            return evals
        finally:
            # モデルを保存
            tk.models.save(model, model_path)


def evaluate(
    model: keras.models.Model,
    dataset: tk.data.Dataset,
    preprocessor: tk.data.Preprocessor,
    batch_size: int = 32,
    prefix: str = "",
    use_horovod: bool = False,
) -> dict:
    """評価して結果をINFOログ出力する。

    Args:
        model: モデル
        dataset: データ
        preprocessor: 前処理
        batch_size: バッチサイズ
        prefix: メトリクス名の接頭文字列。
        use_horovod: MPIによる分散処理をするか否か。

    Returns:
        metricsの文字列と値のdict

    """
    evals = tk.models.evaluate(
        model,
        dataset,
        preprocessor=preprocessor,
        batch_size=batch_size,
        prefix=prefix,
        use_horovod=use_horovod,
    )
    if tk.hvd.is_master():
        max_len = max([len(n) for n in evals])
        for n, v in evals.items():
            tk.log.get(__name__).info(f'{n}:{" " * (max_len - len(n))} {v:.3f}')
    tk.hvd.barrier()
    return evals
