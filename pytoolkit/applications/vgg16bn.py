"""VGG16にBNを加えたもの。"""


def vgg16bn(include_top=False, input_shape=None, input_tensor=None, weights='imagenet'):
    """VGG16にBNを加えたもの。"""
    import keras

    if input_shape is None:
        assert input_tensor is not None
    else:
        assert input_tensor is None
        input_tensor = keras.layers.Input(input_shape)
    assert not include_top  # Trueは未対応
    assert weights in (None, 'imagenet')

    reg = keras.regularizers.l2(1e-4)

    def _conv2d(filters, name):
        def _layer(x):
            x = keras.layers.Conv2D(filters, 3, use_bias=False, padding='same', kernel_regularizer=reg, name=name)(x)
            x = keras.layers.BatchNormalization(gamma_regularizer=reg, beta_regularizer=reg, name=f'{name}_bn')(x)
            x = keras.layers.Activation('relu', name=f'{name}_act')(x)
            return x
        return _layer

    def _pool2d(name):
        return keras.layers.MaxPooling2D(name=name)

    x = _conv2d(64, name='block1_conv1')(input_tensor)
    x = _conv2d(64, name='block1_conv2')(x)
    x = _pool2d(name='block1_pool')(x)
    x = _conv2d(128, name='block2_conv1')(x)
    x = _conv2d(128, name='block2_conv2')(x)
    x = _pool2d(name='block2_pool')(x)
    x = _conv2d(256, name='block3_conv1')(x)
    x = _conv2d(256, name='block3_conv2')(x)
    x = _conv2d(256, name='block3_conv3')(x)
    x = _pool2d(name='block3_pool')(x)
    x = _conv2d(512, name='block4_conv1')(x)
    x = _conv2d(512, name='block4_conv2')(x)
    x = _conv2d(512, name='block4_conv3')(x)
    x = _pool2d(name='block4_pool')(x)
    x = _conv2d(512, name='block5_conv1')(x)
    x = _conv2d(512, name='block5_conv2')(x)
    x = _conv2d(512, name='block5_conv3')(x)
    x = _pool2d(name='block5_pool')(x)

    inputs = keras.engine.get_source_inputs(input_tensor)
    model = keras.models.Model(inputs, x, name='vgg16bn')

    if weights == 'imagenet':
        base_model = keras.applications.VGG16(include_top=False, weights='imagenet')
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                layer.set_weights(base_model.get_layer(name=layer.name).get_weights()[:1])  # biasは捨てる (bnに適切に設定するのがより良いが、とりあえず)

    return model


def preprocess_input(x):
    """前処理。"""
    import keras
    return keras.applications.vgg16.preprocess_input(x)
