# pylint: skip-file
"""DeepLearning(主にKeras)関連。

kerasをimportしてしまうとTensorFlowの初期化が始まって重いので、
importしただけではkerasがimportされないように作っている。

"""

from .dl import *
from . import metrics
