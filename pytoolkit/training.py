"""Kerasでの学習周りの便利関数など。"""
import contextlib

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
    create_model_fn,
    train_set,
    folds,
    train_preprocessor,
    val_preprocessor,
    batch_size=32,
    *,
    models_dir,
    model_name_format="model.fold{fold}.h5",
    skip_if_exists=True,
    **kwargs,
):
    """CV。

    モデルが存在すればスキップしつつfoldsに従ってcross validationする。

    1/5 foldだけ実行したいとかがあればfoldsの要素を先頭1個にすればOK。

    Args:
        create_model_fn (callable): モデルを作成する関数。
        train_set (tk.data.Dataset): 訓練データ
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        folds (array-like): train/valのindexの配列のtupleの配列。(sklearn.model_selection.KFold().split()の結果など)
        batch_size (int): バッチサイズ
        models_dir (PathLike object): モデルの保存先パス (必須)
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists (bool): モデルが存在してもスキップせず再学習するならFalse。
        kwargs (dict): tk.models.fit()のパラメータ

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
    dataset,
    preprocessor,
    batch_size=32,
    load_model_fn=None,
    *,
    nfold=None,
    models=None,
    models_dir=None,
    model_name_format="model.fold{fold}.h5",
    oof=False,
    folds=None,
    use_horovod=False,
    **kwargs,
):
    """CVで作ったモデルで予測。

    Args:
        dataset (tk.data.Dataset): データ
        preprocessor (tk.data.Preprocessor): 前処理
        batch_size (int): バッチサイズ
        load_model_fn (callable): モデルを読み込む関数
        nfold (int): CVのfold数 (models, folds未指定時必須)
        models (list of keras.models.Model): 各foldのモデル
        models_dir (PathLike object): モデルの保存先パス (models未指定時必須)
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        oof (bool): out-of-fold predictionならTrueにする。folds必須。
        folds (array-like): oof時のみ指定する。train/valのindexの配列のtupleの配列。(sklearn.model_selection.KFold().split()の結果など)
        use_horovod (bool): tk.models.predictの引数
        kwargs (dict): tk.models.predictの引数

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
            folds = [(range(0), range(len(dataset))) for _ in range(nfold)]
        load_model_fn = load_model_fn or tk.models.load

        with tk.dl.session(
            use_horovod=use_horovod
        ) if models is None else contextlib.nullcontext():
            if models is None:
                models = [
                    load_model_fn(models_dir / model_name_format.format(fold=fold))
                    for fold in tk.utils.trange(nfold, desc="load models")
                ]

            pred_list = []
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
    train_set,
    train_preprocessor,
    val_set=None,
    val_preprocessor=None,
    batch_size=32,
    *,
    model_path,
    **kwargs,
):
    """学習。

    Args:
        model (keras.models.Model): モデル
        train_set (tk.data.Dataset): 訓練データ
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_set (tk.data.Dataset): 検証データ
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        batch_size (int): バッチサイズ
        model_path (PathLike object): モデルの保存先パス (必須)
        kwargs (dict): tk.models.fit()のパラメータ

    Returns:
        dict: val_setがNoneでなければevaluate結果 (metricsの文字列と値のdict)

    """
    with tk.log.trace_scope("train"):
        assert model_path is not None
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
            if val_set:
                evals = evaluate(
                    model,
                    val_set,
                    preprocessor=val_preprocessor,
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
    model: keras.models.Model,
    dataset,
    preprocessor=None,
    batch_size=32,
    prefix="",
    use_horovod=False,
):
    """評価して結果をINFOログ出力する。

    Args:
        model (keras.models.Model): モデル
        dataset (tk.data.Dataset): データ
        preprocessor (tk.data.Preprocessor): 前処理
        batch_size (int): バッチサイズ
        prefix (str): `""` or `"val_"`
        use_horovod (bool): MPIによる分散処理をするか否か。

    Returns:
        dict: metricsの文字列と値のdict

    """
    evals = tk.models.evaluate(
        model,
        dataset,
        preprocessor=preprocessor,
        batch_size=batch_size,
        use_horovod=use_horovod,
    )
    if tk.hvd.is_master():
        max_len = max([len(n) for n in evals]) + max(len(prefix), 4)
        for n, v in evals.items():
            tk.log.get(__name__).info(
                f'{prefix}{n}:{" " * (max_len - len(prefix) - len(n))} {v:.3f}'
            )
    tk.hvd.barrier()
    return evals
