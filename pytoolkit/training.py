"""Kerasでの学習周りの便利関数など。"""

from . import data, hvd, keras, log, models

_logger = log.get(__name__)


@log.trace()
def check(model: keras.models.Model, plot_path=None):
    """モデルの動作確認など。

    Args:
        model (keras.models.Model): モデル
        plot_path (PathLike object): モデルのplotの保存先パス

    """
    # summary表示
    models.summary(model)
    # グラフを出力
    if plot_path is not None:
        models.plot(model, plot_path)


@log.trace()
def train(model: keras.models.Model, train_dataset: data.Dataset, val_dataset: data.Dataset = None,
          batch_size=16, model_path=None, **kwargs):
    """学習。

    Args:
        model (keras.models.Model): モデル
        train_dataset (data.Dataset): 訓練データ
        val_dataset (data.Dataset): 検証データ
        batch_size (int): バッチサイズ
        model_path (PathLike object): モデルの保存先パス (必須)
        kwargs (dict): models.fitのパラメータ

    """
    assert model_path is not None
    # 学習
    hvd.barrier()
    models.fit(model, train_dataset, validation_data=val_dataset, batch_size=batch_size, **kwargs)
    try:
        # 評価
        evaluate(model, train_dataset, batch_size=batch_size, prefix='', use_horovod=True)
        if val_dataset:
            evaluate(model, val_dataset, batch_size=batch_size, prefix='val_', use_horovod=True)
    finally:
        # モデルを保存
        models.save(model, model_path)


def evaluate(model: keras.models.Model, dataset: data.Dataset, batch_size, prefix, use_horovod=False):
    """評価して結果をINFOログ出力する。

    Args:
        model (keras.models.Model): モデル
        dataset (data.Dataset): データ
        batch_size (int): バッチサイズ
        prefix (str): '' or 'val_'
        use_horovod (bool): MPIによる分散処理をするか否か。

    """
    evals = models.evaluate(model, dataset, batch_size=batch_size, use_horovod=use_horovod)
    if hvd.is_master():
        max_len = max([len(n) for n in evals]) + max(len(prefix), 4)
        for n, v in evals.items():
            _logger.info(f'{prefix}{n}:{" " * (max_len - len(prefix) - len(n))} {v:.3f}')
    hvd.barrier()
