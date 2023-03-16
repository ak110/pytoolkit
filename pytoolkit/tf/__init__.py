"""API。"""
# pylint: skip-file
# flake8: noqa

import tensorflow as tf

from . import layers, tdnn

# ちょっとお行儀が悪いが、logging使ってる前提で設定してしまう
tf.get_logger().propagate = False
del tf
