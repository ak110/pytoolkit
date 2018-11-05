"""セマンティックセグメンテーションを簡単にまとめたコード。"""
import pathlib
import typing

import numpy as np

from . import hvd, layers, losses, metrics, models, networks
from .. import applications, generator, image, jsonex, log, math, ndimage, utils


class SemanticSegmentor(models.Model):
    """セマンティックセグメンテーション。"""

    @classmethod
    def create(cls, class_colors=None, void_color=None, input_size=256, batch_size=16,
               rotation_type='all', color_jitters=True, random_erasing=True,
               weights='imagenet'):
        """学習用インスタンスの作成。

        # 引数
        - class_colors: クラスの色の配列 or None (Noneなら白黒の2クラス)
        - void_color: ラベル付けされていないピクセルがある場合、その色

        """
        assert class_colors is None or len(class_colors) >= 3
        num_classes = 2 if class_colors is None else len(class_colors)

        import keras
        builder = networks.Builder()
        inputs = [builder.input_tensor((None, None, 3))]
        x = inputs[0]
        x = x_in = builder.preprocess('div255')(x)
        # encoder
        base_network = applications.darknet53.darknet53(
            include_top=False, input_tensor=x, weights=weights)
        lr_multipliers = {l: 0.1 for l in base_network.layers}
        down_list = [x_in]  # stage 0: 1/1
        down_list.append(base_network.get_layer(name='add_1').output)  # stage 1: 1/2
        down_list.append(base_network.get_layer(name='add_3').output)  # stage 2: 1/4
        down_list.append(base_network.get_layer(name='add_11').output)  # stage 3: 1/8
        down_list.append(base_network.get_layer(name='add_19').output)  # stage 4: 1/16
        down_list.append(base_network.get_layer(name='add_23').output)  # stage 5: 1/32
        x = base_network.outputs[0]
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = builder.dense(128)(x)
        x = builder.act()(x)
        x = builder.dense(512)(x)
        x = keras.layers.Reshape((1, 1, 512))(x)
        # decoder
        for stage, (d, filters) in list(enumerate(zip(down_list, [16, 32, 64, 128, 256, 512])))[::-1]:
            if stage != len(down_list) - 1:
                x = layers.subpixel_conv2d()(scale=2)(x)
                x = builder.conv2d(filters, 1, use_act=False)(x)
            d = builder.scse_block(builder.shape(d)[-1])(d)
            d = builder.conv2d(filters, 1, use_act=False)(d)
            x = keras.layers.add([x, d])
            x = builder.res_block(filters)(x)
            x = builder.res_block(filters)(x)
            x = builder.bn_act()(x)
        # output
        if num_classes == 2:
            x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
            loss = losses.lovasz_hinge_elup1
            mets = [metrics.binary_accuracy]
            assert void_color is None
        else:
            x = builder.conv2d(num_classes, use_bias=True, use_bn=False, activation='softmax')(x)
            loss = losses.make_lovasz_softmax(ignore=num_classes)
            mets = ['acc']

        network = keras.models.Model(inputs, x)
        gen = _create_generator(class_colors, void_color, (input_size, input_size), rotation_type=rotation_type,
                                color_jitters=color_jitters, random_erasing=random_erasing)
        model = cls(network, gen, batch_size,
                    class_colors=class_colors, void_color=void_color,
                    input_size=input_size, rotation_type=rotation_type)
        model.compile(sgd_lr=0.1 / 128, loss=loss, metrics=mets,
                      lr_multipliers=lr_multipliers, clipnorm=10.0)
        if weights in (None, 'imagenet'):
            pass  # cold start
        else:
            log.get(__name__).info(f'Warm start: {weights}')
            model.load_weights(weights)
        return model

    @classmethod
    def load(cls, filepath: typing.Union[str, pathlib.Path], batch_size=16):  # pylint: disable=W0221
        """予測用インスタンスの作成。"""
        filepath = pathlib.Path(filepath)
        # メタデータの読み込み
        metadata = jsonex.load(filepath.with_suffix('.json'))
        class_colors = metadata['class_colors']
        void_color = metadata['void_color']
        input_size = int(metadata.get('input_size', 256))
        rotation_type = metadata.get('rotation_type', 'none')
        gen = _create_generator(class_colors, void_color, (input_size, input_size))
        # モデルの読み込み
        network = models.load_model(filepath, compile=False)
        # 1回予測して計算グラフを構築
        network.predict_on_batch(np.zeros((1, input_size, input_size, 3)))
        logger = log.get(__name__)
        logger.info('trainable params: %d', models.count_trainable_params(network))
        model = cls(network, gen, batch_size,
                    class_colors=class_colors, void_color=void_color,
                    input_size=input_size, rotation_type=rotation_type)
        return model

    def __init__(self, network, gen, batch_size, postprocess=None,
                 class_colors=None, void_color=None, input_size=None, rotation_type=None):
        super().__init__(network, gen, batch_size, postprocess=postprocess)
        self.class_colors = class_colors
        self.void_color = void_color
        self.input_size = input_size
        self.rotation_type = rotation_type

    def save(self, filepath: typing.Union[str, pathlib.Path], overwrite=True, include_optimizer=True):
        """保存。"""
        filepath = pathlib.Path(filepath)
        # メタデータの保存
        if hvd.is_master():
            metadata = {
                'class_colors': self.class_colors,
                'void_color': self.void_color,
                'rotation_type': self.rotation_type,
            }
            jsonex.dump(metadata, filepath.with_suffix('.json'))
        # モデルの保存
        super().save(filepath, overwrite=overwrite, include_optimizer=include_optimizer)

    def compute_mean_iou(self, y_true, y_pred):
        """クラス毎のIoUを算出する。"""
        num_classes = y_pred.shape[-1]
        i2o = make_image_to_onehot(self.class_colors, self.void_color)
        inters = np.zeros((num_classes,))
        unions = np.zeros((num_classes,))
        for yt_path, yp in zip(y_true, utils.tqdm(y_pred, desc='compute_mean_iou')):
            # 答えを読み込む＆予測結果をリサイズしてクラス化
            if self.class_colors is None:
                yt = i2o(ndimage.load(yt_path)).round()
                yp = ndimage.resize(yp, yt.shape[1], yt.shape[0]).round()
            else:
                yt = i2o(ndimage.load(yt_path))
                mask = np.ravel(yt.sum(axis=-1)) > 0.5  # void_color部分は無視
                yp = np.ravel(ndimage.resize(yp, yt.shape[1], yt.shape[0]).argmax(axis=-1))[mask]
                yt = np.ravel(yt.argmax(axis=-1))[mask]
            # クラスごとに集計
            for c in range(num_classes):
                ct, cp = yt == c, yp == c
                inters[c] += np.sum(np.logical_and(ct, cp))
                unions[c] += np.sum(np.logical_or(ct, cp))
        ious = inters / np.maximum(unions, 1)
        return ious, np.mean(ious)

    def plot(self, filepath, x, pred):
        """予測結果を画像化して保存。"""
        img = ndimage.load(x)
        pred = ndimage.resize(pred, img.shape[1], img.shape[0])
        if self.class_colors is not None:
            colors_table = np.reshape(self.class_colors, (1, 1, len(self.class_colors), 3))
            pred = np.sum(np.expand_dims(pred, axis=-1) * colors_table, axis=-2)
        ndimage.save(filepath, pred)


def _create_generator(class_colors, void_color, image_size,
                      rotation_type='none', color_jitters=False, random_erasing=False):
    """Generatorを作って返す。"""
    gen = image.ImageDataGenerator()
    gen.add(image.LoadOutputImage(grayscale=class_colors is None))
    gen.add(image.Resize(image_size, with_output=True))
    gen.add(image.Padding(probability=1, with_output=True))
    if rotation_type in ('rotation', 'all'):
        gen.add(image.RandomRotate(probability=0.25, degrees=180, with_output=True))
    else:
        gen.add(image.RandomRotate(probability=0.25, with_output=True))
    gen.add(image.RandomCrop(probability=1, with_output=True))
    gen.add(image.Resize(image_size, with_output=True))
    if rotation_type in ('mirror', 'all'):
        gen.add(image.RandomFlipLR(probability=0.5, with_output=True))
    if color_jitters:
        gen.add(image.RandomColorAugmentors())
    if random_erasing:
        gen.add(image.RandomErasing(probability=0.5))
    gen.add(generator.ProcessOutput(make_image_to_onehot(class_colors, void_color)))
    return gen


def make_image_to_onehot(class_colors=None, void_color=None):
    """色をクラスに変換する処理を返す。"""
    if class_colors is None:
        return lambda y: y / 255
    else:
        colors = class_colors[:]
        if void_color is not None:
            assert void_color not in colors
            colors.append(void_color)
        num_classes = len(colors)
        colors_table = np.swapaxes(colors, 0, 1)[np.newaxis, np.newaxis, ...]
        assert colors_table.shape == (1, 1, 3, num_classes)

        def image_to_onehot(y):
            d = np.expand_dims(y, axis=-1) - colors_table
            y = np.negative(np.sum(np.square(d), axis=-2))
            assert y.shape == y.shape[:2] + (num_classes,)
            if void_color is not None:
                y = y[..., :-1]  # void color => all zero
            return math.softmax(y)
        return image_to_onehot
