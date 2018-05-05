"""Kerasのモデル関連。"""

from . import callbacks, hvd, optimizers
from .. import draw, generator, log, utils


class Model(object):
    """`keras.models.Model` + `tk.dl.generators.Generator`の薄いラッパー。"""

    def __init__(self, model, gen: generator.Generator, batch_size):
        import keras
        self.model: keras.models.Model = model
        self.gen = gen
        self.batch_size = batch_size

    def load_weights(self, filepath, where_fn=None):
        """重みの読み込み。

        model.load_weights()は重みの形が違うと読み込めないが、
        警告を出しつつ読むようにしたもの。
        """
        load_weights(self.model, filepath, where_fn)

    def set_multi_gpu_model(self, gpus=None):
        """マルチGPU化。"""
        self.model, self.batch_size = multi_gpu_model(self.model, self.batch_size, gpus)

    @log.trace()
    def compile(self, optimizer=None, loss=None, metrics=None,
                sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
                device_dense='', device_sparse='',
                sgd_lr=None, lr_multipliers=None):
        """コンパイル。

        sgd_lrを指定するとSGD+Nesterov momentumでoptimizerを作る。
        この値はバッチサイズとHorovodを考慮してない値。

        """
        import keras
        assert (optimizer is None) != (sgd_lr is None)  # どちらか必須
        assert loss is not None  # 必須 (optimizerを省略できるようにNoneにしているだけ)
        if lr_multipliers is not None:
            assert sgd_lr is not None

        if sgd_lr is not None:
            if hvd.initialized():
                sgd_lr *= hvd.get().size()
            lr = sgd_lr * self.batch_size
            log.get(__name__).info(f'lr = {lr:.2e}')
            if lr_multipliers is None:
                optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
            else:
                optimizer = optimizers.nsgd()(lr=lr, lr_multipliers=lr_multipliers)

        if hvd.initialized():
            optimizer = hvd.get().DistributedOptimizer(
                optimizer, device_dense=device_dense, device_sparse=device_sparse)

        self.model.compile(optimizer, loss, metrics=metrics,
                           sample_weight_mode=sample_weight_mode,
                           weighted_metrics=weighted_metrics,
                           target_tensors=target_tensors)

    def summary(self):
        """サマリ表示。"""
        print_fn = log.get(__name__).info if hvd.is_master() else lambda _: None
        self.model.summary(print_fn=print_fn)
        print_fn(f'network depth: {count_network_depth(self.model)}')

    @log.trace()
    def fit(self, X_train, y_train,
            epochs=1, verbose=1,
            validation_data: tuple = None,
            class_weight=None,
            max_queue_size=10, use_multiprocessing=False,
            initial_epoch=0,
            tsv_log_path=None,
            balanced=False, mixup=False):
        """学習。

        # 引数
        - tsv_log_path: lossなどをtsvファイルに出力するならそのパス。
        - balanced: クラス間のバランスが均等になるようにオーバーサンプリングするか否か。
        - mixup: Data augmentationにmixupを使用するか否か。

        """
        import keras

        # generatorの用意
        has_val = validation_data is not None
        X_val, y_val = validation_data if has_val else (None, None)
        g1, steps1 = self.gen.flow(X_train, y_train, batch_size=self.batch_size, data_augmentation=True, shuffle=True, balanced=balanced)
        if mixup:
            g1t, steps1t = self.gen.flow(X_train, y_train, batch_size=self.batch_size, data_augmentation=True, shuffle=True, balanced=balanced)
            assert steps1 == steps1t
            g1 = generator.mixup(g1, g1t)
        shuffle_val = hvd.initialized() or balanced
        g2, steps2 = self.gen.flow(X_val, y_val, batch_size=self.batch_size, shuffle=shuffle_val, balanced=balanced) if has_val else (None, None)

        # horovod使用時のsteps per epochの調整
        if hvd.initialized():
            hvd_size = hvd.get().size()
            steps1 //= hvd_size
            if steps1 <= 0:  # 安全装置
                steps1 = 1
            if steps2 is not None:
                steps2 //= hvd_size  # Horovodのサンプルでは * 3 だけど早く進んで欲しいので省略
                if steps2 <= 0:  # 安全装置
                    steps2 = 1

        # callback
        cb = []
        cb.append(callbacks.learning_rate(reduce_epoch_rates=(0.5, 0.75), factor=0.1))
        if hvd.initialized():
            cb.append(hvd.get().callbacks.BroadcastGlobalVariablesCallback(0))
            cb.append(hvd.get().callbacks.MetricAverageCallback())
            cb.append(hvd.get().callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        if tsv_log_path is not None:
            cb.append(callbacks.tsv_logger(tsv_log_path))
        cb.append(callbacks.epoch_logger())
        cb.append(callbacks.freeze_bn(0.95))
        cb.append(keras.callbacks.TerminateOnNaN())

        # 学習
        hist = self.model.fit_generator(
            g1, steps1, epochs=epochs,
            verbose=verbose if hvd.is_master() else 0,
            callbacks=cb,
            validation_data=g2,
            validation_steps=steps2,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            use_multiprocessing=use_multiprocessing,
            shuffle=False,
            initial_epoch=initial_epoch)
        return hist

    @log.trace()
    def predict(self, X, data_augmentation=False):
        """予測。"""
        g, steps = self.gen.flow(X, batch_size=self.batch_size, data_augmentation=data_augmentation)
        return self.model.predict_generator(g, steps)

    @log.trace()
    def evaluate(self, X, y, data_augmentation=False):
        """評価。"""
        g, steps = self.gen.flow(X, y, batch_size=self.batch_size, data_augmentation=data_augmentation)
        return self.model.evaluate_generator(g, steps)

    @log.trace()
    def save(self, filepath, overwrite=True, include_optimizer=True):
        """pathlib対応＆hvd.is_master()な時のみなsave。"""
        if hvd.is_master():
            self.model.save(str(filepath), overwrite=overwrite, include_optimizer=include_optimizer)

    @log.trace()
    def plot(self, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB'):
        """pathlib対応＆hvd.is_master()な時のみ＆エラー握り潰しな`keras.utils.plot_model`。"""
        if hvd.is_master():
            import keras
            try:
                keras.utils.plot_model(self.model, to_file=str(to_file),
                                       show_shapes=show_shapes, show_layer_names=show_layer_names,
                                       rankdir=rankdir)
            except BaseException:
                log.get(__name__).warning('keras.utils.plot_model失敗', exc_info=True)


@log.trace()
def multi_gpu_model(model, batch_size, gpus=None):
    """複数GPUでデータ並列するモデルを作成する。

    # 参考
    https://github.com/fchollet/keras/issues/2436
    https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    """
    if gpus is None:
        gpus = utils.get_gpu_count()
        log.get(__name__).info(f'gpu count = {gpus}')
    if gpus <= 1:
        return model, batch_size

    assert isinstance(model.inputs, list)
    assert isinstance(model.outputs, list)

    import keras

    parallel_model = keras.utils.multi_gpu_model(model, gpus)

    # Model.saveの置き換え
    # https://github.com/fchollet/keras/issues/2436#issuecomment-294243024
    def _save(self_, *args, **kargs):
        assert self_ is not None  # noqa
        model.save(*args, **kargs)

    def _save_weights(self_, *args, **kargs):
        assert self_ is not None  # noqa
        model.save_weights(*args, **kargs)

    parallel_model.save = type(model.save)(_save, parallel_model)
    parallel_model.save_weights = type(model.save_weights)(_save_weights, parallel_model)

    return parallel_model, batch_size * gpus


def freeze_to_name(model, freeze_end_layer_name, skip_bn=False):
    """指定した名前までのレイヤーをfreezeする。skip_bn=TrueならBNはfreezeしない。"""
    import keras
    for layer in model.layers:
        if layer.name == freeze_end_layer_name:
            break
        if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
            pass
        else:
            layer.trainable = False


@log.trace()
def plot_model_params(model, to_file='model.params.png', skip_bn=True):
    """パラメータ数を棒グラフ化"""
    import keras
    import keras.backend as K
    rows = []
    for layer in model.layers:
        if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
            continue
        pc = sum([K.count_params(p) for p in layer.trainable_weights])
        if pc <= 0:
            continue
        rows.append([layer.name, pc])

    import pandas as pd
    df = pd.DataFrame(data=rows, columns=['name', 'params'])

    with draw.get_lock():
        ax = df.plot(x='name', y='params', kind='barh', figsize=(16, 4 * (len(rows) // 32 + 1)))
        ax.invert_yaxis()
        draw.save(ax, to_file)
        draw.close(ax)


def count_trainable_params(model):
    """modelのtrainable paramsを数える"""
    import keras.backend as K
    return sum([sum([K.count_params(p) for p in layer.trainable_weights]) for layer in model.layers])


def count_network_depth(model):
    """重みを持っている層の数を数える。

    「kernel」を持っているレイヤーを数える。
    ConvやDenseなど。ResNet界隈(?)ではDenseは含めないのでずれてしまうが…。
    """
    count = 0
    for layer in model.layers:
        if hasattr(layer, 'kernel') or hasattr(layer, 'depthwise_kernel'):
            count += 1
        elif hasattr(layer, 'layers'):
            count += count_network_depth(layer)
    return count


@log.trace()
def load_model(filepath, compile=True):  # pylint: disable=W0622
    """モデルの読み込み。

    `keras.models.load_model()` + `tk.dl.get_custom_objects()`
    """
    from . import dl
    import keras
    return keras.models.load_model(filepath, custom_objects=dl.get_custom_objects(), compile=compile)


@log.trace()
def load_weights(model, filepath, where_fn=None):
    """重みの読み込み。

    model.load_weights()は重みの形が違うと読み込めないが、
    警告を出しつつ読むようにしたもの。

    # 引数
    - model: 読み込み先モデル。
    - filepath: モデルのファイルパス。(str or pathlib.Path)
    - where_fn: 読み込むレイヤー名を受け取り、読み込むか否かを返すcallable。
    """
    import h5py
    import keras
    import keras.backend as K
    logger = log.get(__name__)
    with h5py.File(str(filepath), mode='r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        original_keras_version = f.attrs['keras_version'].decode('utf8') if 'keras_version' in f.attrs else '1'
        original_backend = f.attrs['backend'].decode('utf8') if 'backend' in f.attrs else None

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]  # pylint: disable=not-an-iterable

        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            if where_fn is not None and not where_fn(name):
                continue

            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names) == 0:
                continue
            weight_values = [g[weight_name] for weight_name in weight_names]

            try:
                layer = model.get_layer(name=name)
            except ValueError as e:
                logger.warning(str(e))
                continue

            symbolic_weights = layer.weights
            weight_values = keras.engine.topology.preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend)
            if len(weight_values) != len(symbolic_weights):
                logger.warning(f'Layer  # {k} (named "{layer.name}") expects {len(symbolic_weights)} weight(s), but the saved weights have {len(weight_values)} element(s).')
                continue
            is_match_shapes = True
            for s, w in zip(symbolic_weights, weight_values):
                if s.shape != w.shape:
                    logger.warning(f'Layer #{k} (named "{layer.name}") expects {s.shape} weight(s), but the saved weights have {w.shape} element(s).')
                    is_match_shapes = False
                    continue
            if is_match_shapes:
                for s, w in zip(symbolic_weights, weight_values):
                    weight_value_tuples.append((s, w))
        K.batch_set_value(weight_value_tuples)
