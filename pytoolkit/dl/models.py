"""Kerasのモデル関連。"""
import warnings

from .. import generator, log, utils


class Model(object):
    """`keras.models.Model` + `tk.dl.generators.Generator`の薄いラッパー。"""

    def __init__(self, model, gen: generator.Generator, use_horovod: bool = False):
        self.model = model  # type: keras.models.Model
        self.gen = gen
        self.use_horovod = use_horovod

    def load_weights(self, filepath, where_fn=None):
        """重みの読み込み。

        model.load_weights()は重みの形が違うと読み込めないが、
        警告を出しつつ読むようにしたもの。
        """
        load_weights(self.model, filepath, where_fn)

    @log.trace()
    def compile(self, optimizer, loss, metrics=None,
                sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
                device_dense='', device_sparse=''):
        """コンパイル。"""
        if self.use_horovod:
            import horovod.keras as hvd
            optimizer = hvd.DistributedOptimizer(
                optimizer, device_dense=device_dense, device_sparse=device_sparse)
        self.model.compile(optimizer, loss, metrics=metrics,
                           sample_weight_mode=sample_weight_mode,
                           weighted_metrics=weighted_metrics,
                           target_tensors=target_tensors)

    def summary(self):
        """サマリ表示。"""
        print_fn = log.get(__name__).info if self.printable else lambda: None
        self.model.summary(print_fn=print_fn)
        print_fn(f'network depth: {count_network_depth(self.model)}')

    def horovod_callbacks(self):
        """Horovodのコールバック3つをまとめて返すだけのやつ。"""
        assert self.use_horovod
        import horovod.keras as hvd
        return [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
        ]

    def fit(self, X_train, y_train,
            batch_size=32, epochs=1, verbose=1, callbacks=None,
            validation_data: tuple = None,
            class_weight=None, initial_epoch=0,
            balanced=False):
        """学習。"""
        has_val = validation_data is not None
        X_val, y_val = validation_data if has_val else (None, None)
        gen1 = self.gen.flow(X_train, y_train, batch_size=batch_size, data_augmentation=True, shuffle=True, balanced=balanced)
        gen2 = self.gen.flow(X_val, y_val, batch_size=batch_size, shuffle=self.use_horovod) if has_val else None
        steps1 = self.gen.steps_per_epoch(len(X_train), batch_size)
        steps2 = self.gen.steps_per_epoch(len(X_val), batch_size) if has_val else None

        if self.use_horovod:
            import horovod.keras as hvd
            steps1 //= hvd.size()
            steps2 //= hvd.size()  # horovodのサンプルでは * 3 だけど早く進んで欲しいので省略

        hist = self.model.fit_generator(
            gen1, steps1, epochs=epochs,
            verbose=verbose if self.printable else 0,
            callbacks=callbacks,
            validation_data=gen2,
            validation_steps=steps2,
            class_weight=class_weight,
            initial_epoch=initial_epoch)
        return hist

    def predict(self, X, batch_size=32, data_augmentation=False):
        """予測。"""
        gen = self.gen.flow(X, batch_size=batch_size, data_augmentation=data_augmentation)
        steps = self.gen.steps_per_epoch(len(X), batch_size=batch_size)
        return self.model.predict_generator(gen, steps)

    def evaluate(self, X, y, batch_size=32, data_augmentation=False):
        """評価。"""
        gen = self.gen.flow(X, y, batch_size=batch_size, data_augmentation=data_augmentation)
        steps = self.gen.steps_per_epoch(len(X), batch_size=batch_size)
        return self.model.evaluate_generator(gen, steps)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        """pathlib対応なsave。"""
        self.model.save(str(filepath), overwrite=overwrite, include_optimizer=include_optimizer)

    def printable(self):
        """Horovodを使ってるなら`hvd.rank() == 0`の場合、そうでないなら常にTrue。"""
        if self.use_horovod:
            import horovod.keras as hvd
            return hvd.rank() == 0
        return True


@log.trace()
def multi_gpu_model(model, batch_size, gpus=None):
    """複数GPUでデータ並列するモデルを作成する。

    # 参考
    https://github.com/fchollet/keras/issues/2436
    https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    """
    if gpus is None:
        gpus = utils.get_gpu_count()
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
    df.plot(x='name', y='params', kind='barh', figsize=(16, 4 * (len(rows) // 32 + 1)))

    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.savefig(str(to_file))
    plt.close()


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
        if hasattr(layer, 'kernel'):
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

            try:
                layer = model.get_layer(name=name)
            except ValueError as e:
                warnings.warn(str(e))
                continue

            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]

            symbolic_weights = layer.weights
            weight_values = keras.engine.topology.preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend)
            if len(weight_values) != len(symbolic_weights):
                warnings.warn('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' +
                              str(len(symbolic_weights)) + ' weight(s), but the saved weights' +
                              ' have ' + str(len(weight_values)) + ' element(s).')
                continue
            is_match_shapes = True
            for s, w in zip(symbolic_weights, weight_values):
                if s.shape != w.shape:
                    warnings.warn('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' +
                                  str(s.shape) + ' weight(s), but the saved weights' +
                                  ' have ' + str(w.shape) + ' element(s).')
                    is_match_shapes = False
                    continue
            if is_match_shapes:
                for s, w in zip(symbolic_weights, weight_values):
                    weight_value_tuples.append((s, w))
        K.batch_set_value(weight_value_tuples)
