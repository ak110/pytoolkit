# pylint: skip-file

import sys
import tensorflow as tf

# kerasがimport済みならkeras、でなくばtf.kerasを使用
if 'keras' in sys.modules:
    import keras
else:
    keras = tf.keras
K = keras.backend

from . import backend
from . import cache
from . import callbacks
from . import data
from . import datasets
from . import dl
from . import hvd
from . import image
from . import layers
from . import log
from . import losses
from . import metrics
from . import ml
from . import ndimage
from . import od
from . import optimizers
from . import utils
from . import vis


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    custom_objects = {}
    custom_objects.update(layers.get_custom_objects())
    custom_objects.update(optimizers.get_custom_objects())
    return custom_objects
