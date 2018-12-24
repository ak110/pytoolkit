"""セマンティックセグメンテーションを簡単にまとめたコード。"""
import pathlib
import typing

import numpy as np
import sklearn.externals.joblib as joblib

from . import hvd, layers, losses, metrics, models, networks
from .. import applications, generator, image, jsonex, log, math, ndimage, utils


def preprocess_masks(mask_files, cache_dir, class_colors, void_color, input_size=None, compress=False):
    """SemanticSegmentor用の前処理。

    マスク画像をone-hot vector化してファイル保存して保存先パスのリストを返す。
    マルチクラスの場合、one-hot化する計算コストが高いため仕方なく用意した。
    2クラスの場合は不要だが、一応同じように動くようにしておく。

    # 引数
    - mask_files: マスク画像のパスのリスト
    - cache_dir: 保存先ディレクトリのパス
    - class_colors: クラスの色の配列 or None (Noneなら白黒の2クラス)
    - void_color: ラベル付けされていないピクセルがある場合、その色
    - input_size: 入力サイズでリサイズする場合、そのサイズ。int or tuple。tupleは(height, width)。
    - compress: 圧縮の有無。(しない方がちょっと早い)

    # 戻り値
    - 保存先パスのリスト

    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if class_colors is None:
        pass
    else:
        colors_table = np.swapaxes(class_colors, 0, 1)[np.newaxis, np.newaxis, ...]
        assert colors_table.shape == (1, 1, 3, len(class_colors))
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    with utils.tqdm(total=len(mask_files), desc='preprocess_masks') as pbar:
        @joblib.delayed
        def _preprocess(p):
            save_path = cache_dir / f'{p.name}{".npz" if compress else ".npy"}'
            if not save_path.exists():
                # 読み込み＆変換
                mask = ndimage.load(p, dtype=np.uint8)
                if class_colors is None:
                    assert np.all(np.isin(mask, (0, 255)))
                else:
                    oh = np.all(np.expand_dims(mask, axis=-1) == colors_table, axis=-2)
                    unmapped_rgb = mask[np.logical_not(np.any(oh, axis=-1))]
                    if void_color is None:
                        assert len(unmapped_rgb) == 0
                    else:
                        assert np.all(unmapped_rgb == np.reshape(void_color, (1, 3))), f'マスク画像に不正な色が存在: {unmapped_rgb}'
                    mask = np.where(oh, np.uint8(255), np.uint8(0))
                # リサイズ
                if input_size is not None:
                    mask = ndimage.resize(mask, input_size[1], input_size[0])
                # 保存
                if compress:
                    np.savez_compressed(str(save_path), mask)
                else:
                    np.save(str(save_path), mask)
            pbar.update(1)
            return save_path

        with joblib.Parallel(backend='threading', n_jobs=-1) as parallel:
            path_list = parallel([_preprocess(p) for p in mask_files])

    return np.array(path_list)


class SemanticSegmentor(models.Model):
    """セマンティックセグメンテーション。"""

    @classmethod
    def create(cls, class_colors=None, void_color=None, input_size=256, batch_size=16,
               rotation_type='all', color_jitters=True, random_erasing=True,
               additional_stages=0,
               weights='imagenet'):
        """学習用インスタンスの作成。

        # 引数
        - class_colors: クラスの色の配列 or None (Noneなら白黒の2クラス)
        - void_color: ラベル付けされていないピクセルがある場合、その色
        - additional_stage: encoderをより深くするならそのダウンサンプリング回数。(1増やすごとに縦横半分になる。端数があると死ぬので注意)
        - weights: None, 'imagenet', 読み込む重みファイルのパスのいずれか。

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
            include_top=False, input_tensor=x,
            weights=weights if weights in (None, 'imagenet') else None)
        lr_multipliers = {l: 0.1 for l in base_network.layers}
        down_list = [x_in]  # stage 0: 1/1
        down_list.append(base_network.get_layer(name='add_1').output)  # stage 1: 1/2
        down_list.append(base_network.get_layer(name='add_3').output)  # stage 2: 1/4
        down_list.append(base_network.get_layer(name='add_11').output)  # stage 3: 1/8
        down_list.append(base_network.get_layer(name='add_19').output)  # stage 4: 1/16
        down_list.append(base_network.get_layer(name='add_23').output)  # stage 5: 1/32
        x = base_network.outputs[0]
        for stage in range(additional_stages):
            x = builder.conv2d(256, 2, strides=2, use_act=False, name=f'down{stage}_conv')(x)
            x = builder.res_block(256, name=f'down{stage}_b1')(x)
            x = builder.res_block(256, name=f'down{stage}_b2')(x)
            x = builder.bn_act(name=f'down{stage}')(x)
            down_list.append(x)
        x = keras.layers.GlobalAveragePooling2D(name='center_pool')(x)
        x = builder.dense(128, name='center_dense1')(x)
        x = builder.act(name='center_act')(x)
        x = builder.dense(256, name='center_dense2')(x)
        x = keras.layers.Reshape((1, 1, 256), name='center_reshape')(x)
        # decoder
        up_list = []
        for stage, d in list(enumerate(down_list))[::-1]:
            filters = min(16 * 2 ** stage, 256)
            if stage != len(down_list) - 1:
                x = layers.subpixel_conv2d()(name=f'up{stage}_us')(x)
                x = builder.conv2d(filters, 1, use_act=False, name=f'up{stage}_ex')(x)
            d = builder.conv2d(filters, 1, use_act=False, name=f'up{stage}_lt')(d)
            x = keras.layers.add([x, d], name=f'up{stage}_add')
            x = builder.res_block(filters, name=f'up{stage}_b1')(x)
            x = builder.res_block(filters, name=f'up{stage}_b2')(x)
            x = builder.bn_act(name=f'up{stage}')(x)
            up_list.append(builder.conv2d(32, 1, use_act=False, name=f'up{stage}_sq')(x))
        # Hypercolumn
        hc = [layers.resize2d()(scale=2 ** (len(up_list) - i - 1), name=f'hc_resize{i}')(u) for i, u in enumerate(up_list)]
        x = keras.layers.add(hc, name='hc_add')
        # Refinement
        x = builder.res_block(32, name='refine_b1')(x)
        x = builder.res_block(32, name='refine_b2')(x)
        x = builder.bn_act(name='refine')(x)
        # output
        if num_classes == 2:
            x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid', name='predictions')(x)
            loss = _binary_ss_loss
            mets = [metrics.binary_accuracy]
            assert void_color is None
        else:
            x = builder.conv2d(num_classes, use_bias=True, use_bn=False, activation='softmax', name=f'predictions_{num_classes}')(x)
            loss = _multiclass_ss_loss
            mets = ['acc']

        network = keras.models.Model(inputs, x)
        gen = _create_generator(class_colors, void_color, (input_size, input_size), rotation_type=rotation_type,
                                color_jitters=color_jitters, random_erasing=random_erasing)
        model = cls(network, gen, batch_size,
                    class_colors=class_colors, void_color=void_color,
                    input_size=input_size, rotation_type=rotation_type)
        model.compile(sgd_lr=1e-3, loss=loss, metrics=mets,
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
        network.predict(np.zeros((1, input_size, input_size, 3)))
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
                'input_size': self.input_size,
                'rotation_type': self.rotation_type,
            }
            jsonex.dump(metadata, filepath.with_suffix('.json'))
        # モデルの保存
        super().save(filepath, overwrite=overwrite, include_optimizer=include_optimizer)

    def compute_mean_iou(self, y_true, y_pred):
        """クラス毎のIoUとその平均(mean IoU)を算出する。

        # 戻り値
        - ious: クラスごとのIoU
        - miou: iousの平均 (ただし2クラスの場合は0:背景、1:物体と見なして物体のIoU)

        """
        if self.class_colors is None:
            num_classes = 2
        else:
            num_classes = y_pred.shape[-1]
        i2o = make_image_to_onehot(self.class_colors, self.void_color)
        inters = np.zeros((num_classes,))
        unions = np.zeros((num_classes,))
        for yt_path, yp in utils.tqdm(list(zip(y_true, y_pred)), desc='mIoU'):
            yt, yp = self.get_mask_for_evaluation(yt_path, yp, i2o)
            # クラスごとに集計
            for c in range(num_classes):
                ct, cp = yt == c, yp == c
                inters[c] += np.sum(np.logical_and(ct, cp))
                unions[c] += np.sum(np.logical_or(ct, cp))
        ious = inters / np.maximum(unions, 1)
        if self.class_colors is None:
            return ious, ious[1]
        else:
            return ious, np.mean(ious)

    def compute_mean_iou_per_image(self, y_true, y_pred):
        """画像ごとのmean IoUを算出する。"""
        i2o = make_image_to_onehot(self.class_colors, self.void_color)
        mious = np.empty((len(y_true),))
        for i, (yt_path, yp) in utils.tqdm(list(enumerate(zip(y_true, y_pred))), desc='mIoU/image'):
            yt, yp = self.get_mask_for_evaluation(yt_path, yp, i2o)
            if self.class_colors is None:
                ct, cp = yt == 1, yp == 1
                inter = np.sum(np.logical_and(ct, cp))
                union = np.sum(np.logical_or(ct, cp))
                mious[i] = inter / union  # IoU (class 1)
            else:
                iou_list = []
                for c in range(len(self.class_colors)):
                    ct, cp = yt == c, yp == c
                    union = np.sum(np.logical_or(ct, cp))
                    if union > 0:
                        inter = np.sum(np.logical_and(ct, cp))
                        iou_list.append(inter / union)
                mious[i] = np.mean(iou_list)  # mean IoU
        return mious

    def plot_mask(self, x, pred, color_mode='soft'):
        """予測結果を画像化して返す。"""
        assert color_mode in ('soft', 'hard')
        _, pred = self.get_mask(x, pred)
        if self.class_colors is None:
            if color_mode == 'soft':
                pred *= 255
            else:
                pred = pred.round() * 255
        else:
            if color_mode == 'soft':
                colors_table = np.reshape(self.class_colors, (1, 1, len(self.class_colors), 3))
                pred = np.sum(np.expand_dims(pred, axis=-1) * colors_table, axis=-2)
            else:
                colors_table = np.array(self.class_colors)
                pred = colors_table[pred.argmax(axis=-1)]
        return pred

    def get_mask_for_evaluation(self, y, pred, i2o):
        """答えと予測結果のマスクを評価用に色々整えて返す。"""
        yt, yp = self.get_mask(y, pred)
        if self.class_colors is None:
            yt = i2o(yt).round().astype(np.uint8)
            yp = yp.round().astype(np.uint8)
        else:
            yt = i2o(yt)
            mask = yt.sum(axis=-1) > 0.5  # void_color部分(all 0)は無視
            yp = yp.argmax(axis=-1)[mask]
            yt = yt.argmax(axis=-1)[mask]
        return yt, yp

    def get_mask(self, x_or_y, pred):
        """予測結果を入力のサイズに合わせて返す。"""
        img = ndimage.load(x_or_y, grayscale=self.class_colors is None)
        pred = ndimage.resize(pred, img.shape[1], img.shape[0])
        return img, pred


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
    return gen


def make_image_to_onehot(class_colors=None, void_color=None, strict=False):
    """色をクラスに変換する処理を返す。"""
    if class_colors is None:
        # binary
        if strict:
            assert np.isin(y, (0, 255))
        return lambda y: y / 255

    # multiclass
    num_classes = len(class_colors)
    colors = class_colors[:]
    if void_color is not None:
        assert void_color not in colors
        colors.append(void_color)
    colors_table = np.swapaxes(colors, 0, 1)[np.newaxis, np.newaxis, ...]
    assert colors_table.shape == (1, 1, 3, len(colors))

    if strict:
        def image_to_onehot(y):
            oh = np.all(np.expand_dims(y, axis=-1) == colors_table, axis=-2).astype(np.float32)
            if void_color is None:
                assert not np.any(np.all(oh == 0, axis=-1)), f'不正な色のピクセルが存在: {y[np.all(oh == 0, axis=-1), :]}'
            return oh
        return image_to_onehot
    else:
        colors_table = colors_table.astype(np.float32)

        def image_to_onehot(y):
            d = np.expand_dims(y, axis=-1).astype(np.float32) - colors_table
            y = np.negative(np.sum(np.square(d), axis=-2))
            assert y.shape == y.shape[:2] + (len(colors),)
            if void_color is not None:
                y = y[..., :-1]  # void color => all zero
            assert y.shape == y.shape[:2] + (num_classes,)
            return math.softmax(y)
        return image_to_onehot


def _binary_ss_loss(y_true, y_pred):
    """2クラス用loss。"""
    import keras
    y_true = y_true / 255  # [0-1)
    loss1 = losses.lovasz_hinge_elup1(y_true, y_pred)
    loss2 = keras.losses.binary_crossentropy(y_true, y_pred)
    return loss1 * 0.9 + keras.backend.clip(loss2, -10, +10) * 0.1


def _multiclass_ss_loss(y_true, y_pred):
    """多クラス用loss。"""
    import keras
    scale = keras.backend.sum(y_true, axis=-1)
    mask = keras.backend.cast(scale > 0, keras.backend.floatx())  # all-zero部分は学習しない
    y_true = y_true / keras.backend.expand_dims(scale, axis=-1)  # 合計を1に補正 (リサイズなどでずれる可能性があるので)
    cce = keras.backend.categorical_crossentropy(y_true, y_pred)
    return cce * mask
