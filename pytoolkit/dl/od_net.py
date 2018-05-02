"""お手製Object detectionのネットワーク部分。"""

import numpy as np

from . import layers, losses, models
from .. import image, log


def get_preprocess_input(base_network):
    """`preprocess_input`を返す。"""
    if base_network in ('vgg16', 'resnet50'):
        return image.preprocess_input_mean
    else:
        assert base_network in ('custom', 'xception')
        return image.preprocess_input_abs1


@log.trace()
def create_network(base_network, pb, mode, strict_nms=None):
    """学習とか予測とか用のネットワークを作って返す。"""
    assert mode in ('pretrain', 'train', 'predict')
    import keras
    builder = layers.Builder()
    x = inputs = keras.layers.Input(pb.input_size + (3,))
    x, ref, lr_multipliers = _create_basenet(base_network, builder, x, load_weights=mode != 'predict')
    if mode == 'pretrain':
        assert len(lr_multipliers) == 0
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = builder.dense(4, activation='softmax', name='predictions')(x)
        model = keras.models.Model(inputs=inputs, outputs=x)
    else:
        for_predict = mode == 'predict'
        model = _create_detector(pb, builder, inputs, x, ref, lr_multipliers, for_predict, strict_nms)

    logger = log.get(__name__)
    logger.info('network depth: %d', models.count_network_depth(model))
    logger.info('trainable params: %d', models.count_trainable_params(model))
    return model, lr_multipliers


@log.trace()
def _create_basenet(base_network, builder, x, load_weights):
    """ベースネットワークの作成。"""
    import keras
    basenet = None
    ref_list = []
    if base_network == 'custom':
        x = builder.conv2d(32, 7, strides=2, name='stage0_ds')(x)
        x = builder.conv2d(64, strides=2, name='stage2_ds')(x)
        x = builder.conv2d(64, strides=1, name='stage2_conv1')(x)
        x = builder.dwconv2d(name='stage2_conv2')(x)
        x = builder.conv2d(128, strides=2, name='stage3_ds')(x)
        x = builder.conv2d(128, strides=1, name='stage3_conv1')(x)
        x = builder.dwconv2d(name='stage3_conv2')(x)
        x = builder.conv2d(256, strides=2, name='stage4_ds')(x)
        x = builder.conv2d(256, strides=1, name='stage4_conv1')(x)
        x = builder.dwconv2d(name='stage4_conv2')(x)
        ref_list.append(x)
    elif base_network == 'vgg16':
        basenet = keras.applications.VGG16(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
        ref_list.append(basenet.get_layer(name='block4_pool').input)
        ref_list.append(basenet.get_layer(name='block5_pool').input)
    elif base_network == 'resnet50':
        basenet = keras.applications.ResNet50(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
        ref_list.append(basenet.get_layer(name='res4a_branch2a').input)
        ref_list.append(basenet.get_layer(name='res5a_branch2a').input)
        ref_list.append(basenet.get_layer(name='avg_pool').input)
    elif base_network == 'xception':
        basenet = keras.applications.Xception(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
        ref_list.append(basenet.get_layer(name='block4_sepconv1_act').input)
        ref_list.append(basenet.get_layer(name='block13_sepconv1_act').input)
        ref_list.append(basenet.get_layer(name='block14_sepconv2_act').output)
    else:
        assert False

    # 転移学習元部分の学習率は控えめにする
    lr_multipliers = {}
    if basenet is not None:
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [0.01] * len(w)))

    # チャンネル数が多ければここで減らす
    x = ref_list[-1]
    if builder.shape(x)[-1] > 256:
        x = builder.conv2d(256, (1, 1), name='tail_sq')(x)
    assert builder.shape(x)[-1] == 256

    # downsampling
    down_index = 0
    while True:
        down_index += 1
        map_size = builder.shape(x)[1] // 2
        x = builder.conv2d(256, strides=2, name=f'down{down_index}_ds')(x)
        x = builder.conv2d(256, strides=1, name=f'down{down_index}_conv1')(x)
        x = builder.dwconv2d(name=f'down{down_index}_conv2')(x)
        assert builder.shape(x)[1] == map_size
        ref_list.append(x)
        if map_size <= 4 or map_size % 2 != 0:  # 充分小さくなるか奇数になったら終了
            break

    ref = {f'down{builder.shape(x)[1]}': x for x in ref_list}
    return x, ref, lr_multipliers


@log.trace()
def _create_detector(pb, builder, inputs, x, ref, lr_multipliers, for_predict, strict_nms):
    """ネットワークのcenter以降の部分を作る。"""
    import keras
    import keras.backend as K
    map_size = builder.shape(x)[1]

    # center
    x = builder.conv2d(64, (1, 1), name='center_conv1')(x)
    x = builder.conv2d(64, (map_size, map_size), padding='valid', name='center_conv2')(x)

    # upsampling
    up_index = 0
    while True:
        up_index += 1
        in_map_size = builder.shape(x)[1]
        assert map_size % in_map_size == 0, f'map size error: {in_map_size} -> {map_size}'
        up_size = map_size // in_map_size
        x = builder.conv2dtr(64, up_size, strides=up_size, padding='valid', name=f'up{up_index}_us')(x)
        x = builder.conv2d(256, 1, use_act=False, name=f'up{up_index}_ex')(x)
        t = builder.conv2d(256, 1, use_act=False, name=f'up{up_index}_lt')(ref[f'down{map_size}'])
        x = keras.layers.add([x, t], name=f'up{up_index}_mix')
        x = builder.bn_act(name=f'up{up_index}_mix')(x)
        x = keras.layers.Dropout(0.25)(x)
        x = builder.conv2d(256, name=f'up{up_index}_conv1')(x)
        x = builder.dwconv2d(name=f'up{up_index}_conv2')(x)
        ref[f'out{map_size}'] = x

        if pb.map_sizes[0] <= map_size:
            break
        map_size *= 2

    # prediction module
    objs, clfs, locs = _create_pm(pb, builder, ref, lr_multipliers)

    if for_predict:
        model = _create_predict_network(pb, inputs, objs, clfs, locs, strict_nms)
    else:
        assert strict_nms is None
        # ラベル側とshapeを合わせるためのダミー
        dummy_shape = (len(pb.pb_locs), 1)
        dummy = keras.layers.Lambda(K.zeros_like, dummy_shape, name='dummy')(objs)  # objsとちょうどshapeが同じなのでzeros_like。
        # いったんくっつける (損失関数の中で分割して使う)
        outputs = keras.layers.concatenate([dummy, objs, clfs, locs], axis=-1, name='outputs')
        model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model


@log.trace()
def _create_pm(pb, builder, ref, lr_multipliers):
    """Prediction module."""
    import keras

    old_gn, builder.use_gn = builder.use_gn, True

    shared_layers = {}
    shared_layers['pm_layer1'] = builder.conv2d(256, use_act=False, name='pm_conv')
    shared_layers['pm_layer2'] = builder.res_block(256, name='pm_res1')
    shared_layers['pm_layer3'] = builder.res_block(256, name='pm_res2')
    shared_layers['pm_bn_act'] = builder.bn_act(name='pm')
    for pat_ix in range(len(pb.pb_size_patterns)):
        shared_layers[f'pm-{pat_ix}_obj'] = builder.conv2d(
            1, 1,
            kernel_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_initializer=losses.od_bias_initializer(1),
            bias_regularizer=None,
            activation='sigmoid',
            use_bias=True,
            use_bn=False,
            name=f'pm-{pat_ix}_obj')
        shared_layers[f'pm-{pat_ix}_clf'] = builder.conv2d(
            pb.num_classes, 1,
            kernel_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l2(1e-4),
            activation='softmax',
            use_bias=True,
            use_bn=False,
            name=f'pm-{pat_ix}_clf')
        shared_layers[f'pm-{pat_ix}_loc'] = builder.conv2d(
            4, 1,
            kernel_initializer='zeros',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l2(1e-4),
            use_bias=True,
            use_bn=False,
            use_act=False,
            name=f'pm-{pat_ix}_loc')
    for layer in shared_layers.values():
        w = layer.trainable_weights
        lr_multipliers.update(zip(w, [1 / len(pb.map_sizes)] * len(w)))  # 共有部分の学習率調整

    builder.use_gn = old_gn

    objs, clfs, locs = [], [], []
    for map_size in pb.map_sizes:
        assert f'out{map_size}' in ref, f'map_size error: {ref}'
        x = ref[f'out{map_size}']
        x = shared_layers[f'pm_layer1'](x)
        x = shared_layers[f'pm_layer2'](x)
        x = shared_layers[f'pm_layer3'](x)
        x = shared_layers[f'pm_bn_act'](x)
        for pat_ix in range(len(pb.pb_size_patterns)):
            obj = shared_layers[f'pm-{pat_ix}_obj'](x)
            clf = shared_layers[f'pm-{pat_ix}_clf'](x)
            loc = shared_layers[f'pm-{pat_ix}_loc'](x)
            obj = keras.layers.Reshape((-1, 1), name=f'pm{map_size}-{pat_ix}_r1')(obj)
            clf = keras.layers.Reshape((-1, pb.num_classes), name=f'pm{map_size}-{pat_ix}_r2')(clf)
            loc = keras.layers.Reshape((-1, 4), name=f'pm{map_size}-{pat_ix}_r3')(loc)
            objs.append(obj)
            clfs.append(clf)
            locs.append(loc)
    objs = keras.layers.concatenate(objs, axis=-2, name='output_objs')
    clfs = keras.layers.concatenate(clfs, axis=-2, name='output_clfs')
    locs = keras.layers.concatenate(locs, axis=-2, name='output_locs')
    return objs, clfs, locs


def _create_predict_network(pb, inputs, objs, clfs, locs, strict_nms):
    """予測用ネットワークの作成"""
    import keras
    import keras.backend as K
    nms_all_threshold = 0.5 if strict_nms else None

    def _conf(x):
        objs = x[0][:, :, 0]
        confs = x[1]
        # objectnessとconfidenceの幾何平均をconfidenceということにしてみる
        conf = K.sqrt(objs * confs)
        return conf * np.expand_dims(pb.pb_mask, axis=0)

    classes = layers.channel_argmax()(name='classes')(clfs)
    confs = layers.channel_max()(name='confs')(clfs)
    objconfs = keras.layers.Lambda(_conf, K.int_shape(clfs)[1:-1], name='objconfs')([objs, confs])
    locs = keras.layers.Lambda(pb.decode_locs, name='locs')(locs)
    nms = layers.nms()(pb.num_classes, len(pb.pb_locs), nms_all_threshold=nms_all_threshold, name='nms')([classes, objconfs, locs])
    return keras.models.Model(inputs=inputs, outputs=nms)
