"""Kerasでの学習周りの便利関数など。"""

import numpy as np

import pytoolkit as tk

from . import keras
from . import log as tk_log


@tk_log.trace()
def check(model: keras.models.Model, plot_path=None):
    """モデルの動作確認など。

    Args:
        model (keras.models.Model): モデル
        plot_path (PathLike object): モデルのplotの保存先パス

    """
    # summary表示
    tk.models.summary(model)
    # グラフを出力
    if plot_path is not None:
        tk.models.plot(model, plot_path)


@tk_log.trace()
def cv(
    create_model_fn,
    train_dataset,
    folds,
    batch_size=16,
    *,
    models_dir,
    model_name_format="model.{fold}.h5",
    **kwargs,
):
    """CV。

    Args:
        create_model_fn (callable): モデルを作成する関数。
        train_dataset (tk.data.Dataset): 訓練データ
        folds (array-like): train/valのindexの配列のtupleの配列。(sklearn.model_selection.KFold().split()の結果など)
        batch_size (int): バッチサイズ
        models_dir (PathLike object): モデルの保存先パス (必須)
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        kwargs (dict): tk.models.fit()のパラメータ

    Returns:
        dict: evaluate結果 (metricsの文字列と値のdict)

    """
    evals_list = []

    for fold, (train_indices, val_indices) in enumerate(folds):
        tr = tk.data.SubDataset(train_dataset, train_indices)
        vl = tk.data.SubDataset(train_dataset, val_indices)
        model_path = models_dir / model_name_format.format(fold=fold)
        with tk.dl.session(use_horovod=True):
            model = create_model_fn()
            evals = train(
                model, tr, vl, batch_size=batch_size, model_path=model_path, **kwargs
            )
            evals_list.append(evals)

    results = {}
    for k in evals_list[0]:
        results[k] = np.mean([e[k] for e in evals_list])
    return results


@tk_log.trace()
def train(
    model: keras.models.Model,
    train_dataset,
    val_dataset=None,
    batch_size=16,
    *,
    model_path,
    **kwargs,
):
    """学習。

    Args:
        model (keras.models.Model): モデル
        train_dataset (tk.data.Dataset): 訓練データ
        val_dataset (tk.data.Dataset): 検証データ
        batch_size (int): バッチサイズ
        model_path (PathLike object): モデルの保存先パス (必須)
        kwargs (dict): tk.models.fit()のパラメータ

    Returns:
        dict: val_datasetがNoneでなければevaluate結果 (metricsの文字列と値のdict)

    """
    assert model_path is not None
    # 学習
    tk.log.get(__name__).info(
        f"train: {len(train_dataset)} samples, val: {len(val_dataset) if val_dataset is not None else 0} samples, batch_size: {batch_size}x{tk.hvd.size()}"
    )
    tk.hvd.barrier()
    tk.models.fit(
        model,
        train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        **kwargs,
    )
    try:
        # 評価
        evaluate(
            model, train_dataset, batch_size=batch_size, prefix="", use_horovod=True
        )
        if val_dataset:
            evals = evaluate(
                model,
                val_dataset,
                batch_size=batch_size,
                prefix="val_",
                use_horovod=True,
            )
        else:
            evals = None
        return evals
    finally:
        # モデルを保存
        tk.models.save(model, model_path)


def evaluate(
    model: keras.models.Model, dataset, batch_size, prefix="", use_horovod=False
):
    """評価して結果をINFOログ出力する。

    Args:
        model (keras.models.Model): モデル
        dataset (tk.data.Dataset): データ
        batch_size (int): バッチサイズ
        prefix (str): '' or 'val_'
        use_horovod (bool): MPIによる分散処理をするか否か。

    Returns:
        dict: metricsの文字列と値のdict

    """
    evals = tk.models.evaluate(
        model, dataset, batch_size=batch_size, use_horovod=use_horovod
    )
    if tk.hvd.is_master():
        max_len = max([len(n) for n in evals]) + max(len(prefix), 4)
        for n, v in evals.items():
            tk_log.get(__name__).info(
                f'{prefix}{n}:{" " * (max_len - len(prefix) - len(n))} {v:.3f}'
            )
    tk.hvd.barrier()
    return evals
