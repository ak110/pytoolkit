# pylint: skip-file

import tensorflow as tf


def _use_tf_keras():
    """Trueならtf.keras、Falseならkerasを使う。"""
    import os

    backend = os.environ.get("PYTOOLKIT_BACKEND", None)
    if backend == "tf":
        return True
    elif backend == "keras":
        return False
    else:
        import sys

        if "keras" in sys.modules:
            return False
        return True


if True:
    # tf.keras or keras
    if _use_tf_keras():
        from tensorflow import keras
    else:
        print("Using native Keras.")
        import keras
    K = keras.backend

    # その他のimport
    from . import log
    from . import applications
    from . import datasets
    from . import evaluations
    from . import pipeline
    from . import autoaugment
    from . import backend
    from . import cache
    from . import callbacks
    from . import cli
    from . import data
    from . import dl
    from . import hpo
    from . import hvd
    from . import image
    from . import layers
    from . import losses
    from . import math
    from . import metrics
    from . import ml
    from . import models
    from . import ndimage
    from . import notifications
    from . import od
    from . import optimizers
    from . import preprocessing
    from . import table
    from . import threading
    from . import training
    from . import utils
    from . import validation
    from . import vis
    from . import web


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    custom_objects = {}
    custom_objects.update(layers.get_custom_objects())
    custom_objects.update(optimizers.get_custom_objects())
    return custom_objects
