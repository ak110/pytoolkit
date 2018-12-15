"""画像分類の転移学習周りを簡単にまとめたコード。"""
import pathlib
import typing

import numpy as np

from . import hvd, models
from .. import applications, generator, image, jsonex, log, ml


class ImageClassifier(models.Model):
    """画像分類モデル。"""

    @classmethod
    def create(cls, class_names, network_type='nasnet_large', input_size=256, batch_size=16,
               rotation_type='all', color_jitters=True, random_erasing=True,
               freeze_type='none', additional_stages=0, weights='imagenet'):
        """学習用インスタンスの作成。"""
        assert len(class_names) >= 2
        assert network_type in ('vgg16bn', 'resnet50', 'xception', 'darknet53', 'nasnet_large')
        assert batch_size >= 1
        assert rotation_type in ('none', 'mirror', 'rotation', 'all')
        assert freeze_type in ('none', 'without_bn', 'all')
        assert weights in (None, 'imagenet')
        network, preprocess_mode = _create_network(len(class_names), network_type, (input_size, input_size), freeze_type, additional_stages, weights)
        gen = _create_generator(len(class_names), (input_size, input_size), preprocess_mode,
                                rotation_type=rotation_type, color_jitters=color_jitters, random_erasing=random_erasing)
        model = cls(class_names, network_type, preprocess_mode, input_size, rotation_type, network, gen, batch_size)
        model.compile(sgd_lr=0.1 / 128, loss='categorical_crossentropy', metrics=['acc'])
        return model

    @classmethod
    def load(cls, filepath: typing.Union[str, pathlib.Path], batch_size=16):  # pylint: disable=W0221
        """予測用インスタンスの作成。"""
        filepath = pathlib.Path(filepath)
        # メタデータの読み込み
        metadata = jsonex.load(filepath.with_suffix('.json'))
        class_names = metadata['class_names']
        network_type = metadata.get('network_type', None)
        preprocess_mode = metadata['preprocess_mode']
        input_size = int(metadata.get('input_size', 256))
        rotation_type = metadata.get('rotation_type', 'none')
        gen = _create_generator(len(class_names), (input_size, input_size), preprocess_mode)
        # モデルの読み込み
        network = models.load_model(filepath, compile=False)
        # 1回予測して計算グラフを構築
        network.predict_on_batch(np.zeros((1, input_size, input_size, 3)))
        logger = log.get(__name__)
        logger.info('trainable params: %d', models.count_trainable_params(network))
        return cls(class_names, network_type, preprocess_mode, input_size, rotation_type, network, gen, batch_size)

    def __init__(self, class_names, network_type, preprocess_mode, input_size, rotation_type, network, gen, batch_size, postprocess=None):
        super().__init__(network, gen, batch_size, postprocess=postprocess)
        self.class_names = class_names
        self.network_type = network_type
        self.preprocess_mode = preprocess_mode
        self.input_size = input_size
        self.rotation_type = rotation_type

    def save(self, filepath: typing.Union[str, pathlib.Path], overwrite=True, include_optimizer=True):
        """保存。"""
        filepath = pathlib.Path(filepath)
        # メタデータの保存
        if hvd.is_master():
            metadata = {
                'class_names': self.class_names,
                'network_type': self.network_type,
                'preprocess_mode': self.preprocess_mode,
                'input_size': self.input_size,
                'rotation_type': self.rotation_type,
            }
            jsonex.dump(metadata, filepath.with_suffix('.json'))
        # モデルの保存
        super().save(filepath, overwrite=overwrite, include_optimizer=include_optimizer)


def _create_network(num_classes, network_type, image_size, freeze_type, additional_stages, weights):
    """ネットワークを作って返す。"""
    import tensorflow as tf
    if network_type == 'vgg16bn':
        base_model = applications.vgg16bn.vgg16bn(include_top=False, input_shape=(None, None, 3), weights=weights)
        preprocess_mode = 'caffe'
    elif network_type == 'resnet50':
        base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3), weights=weights)
        preprocess_mode = 'caffe'
    elif network_type == 'xception':
        base_model = tf.keras.applications.Xception(include_top=False, input_shape=(None, None, 3), weights=weights)
        preprocess_mode = 'tf'
    elif network_type == 'darknet53':
        base_model = applications.darknet53.darknet53(include_top=False, input_shape=(None, None, 3), weights=weights)
        preprocess_mode = 'div255'
    elif network_type == 'nasnet_large':
        base_model = tf.keras.applications.NASNetLarge(include_top=False, input_shape=image_size + (3,), weights=weights)
        preprocess_mode = 'tf'
    else:
        raise ValueError(f'Invalid network type: {network_type}')

    if freeze_type == 'without_bn':
        for layer in base_model.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    elif freeze_type == 'all':
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.outputs[0]
    if additional_stages > 0:
        x = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l2(1e-4),
                                               beta_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Activation('relu')(x)
    for _ in range(additional_stages):
        x = tf.keras.layers.Conv2D(256, 2, strides=2, padding='same', use_bias=False,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l2(1e-4),
                                               beta_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        for _ in range(3):
            shortcut = x
            x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False,
                                       kernel_initializer='he_uniform',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l2(1e-4),
                                                   beta_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False,
                                       kernel_initializer='he_uniform',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l2(1e-4),
                                                   beta_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = tf.keras.layers.add([shortcut, x])
        x = tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l2(1e-4),
                                               beta_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if freeze_type == 'all' and additional_stages <= 0:
        x = tf.keras.layers.Dense(2048, activation='relu',
                                  kernel_initializer='he_uniform',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                  name=f'pred_{num_classes}_pre')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_initializer='zeros',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name=f'pred_{num_classes}')(x)
    model = tf.keras.models.Model(base_model.inputs, x)
    return model, preprocess_mode


def _create_generator(num_classes, image_size, preprocess_mode, rotation_type='none', color_jitters=False, random_erasing=False):
    """Generatorを作って返す。"""
    gen = image.ImageDataGenerator()
    gen.add(image.Resize(image_size))
    gen.add(image.Padding(probability=1))
    if rotation_type in ('rotation', 'all'):
        gen.add(image.RandomRotate(probability=0.25, degrees=180))
    else:
        gen.add(image.RandomRotate(probability=0.25))
    gen.add(image.RandomCrop(probability=1))
    gen.add(image.Resize(image_size))
    if rotation_type in ('mirror', 'all'):
        gen.add(image.RandomFlipLR(probability=0.5))
    if color_jitters:
        gen.add(image.RandomColorAugmentors())
    if random_erasing:
        gen.add(image.RandomErasing(probability=0.5))
    gen.add(image.Preprocess(mode=preprocess_mode))
    gen.add(generator.ProcessOutput(ml.to_categorical(num_classes), batch_axis=True))
    return gen
