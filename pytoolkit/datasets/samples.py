"""tf.keras.datasets関連。 <https://tf.keras.io/datasets/>"""

import numpy as np
import sklearn.datasets

import pytoolkit as tk


def load_sample_od():
    """物体検出のサンプルデータ。num_classes=2"""
    X_train = np.array(sklearn.datasets.load_sample_images().filenames)
    y_train = np.array(
        [
            tk.od.ObjectsAnnotation(
                X_train[0],
                640,
                427,
                classes=[0],
                bboxes=[[61 / 640, 29 / 427, 372 / 640, 427 / 427]],
            ),
            tk.od.ObjectsAnnotation(
                X_train[1],
                640,
                427,
                classes=[1, 1],
                bboxes=[
                    [168 / 640, 85 / 427, 447 / 640, 362 / 427],
                    [286 / 640, 373 / 427, 471 / 640, 427 / 427],
                ],
            ),
        ]
    )
    return tk.data.Dataset(
        X_train, y_train, metadata={"class_names": ["china", "flower"]}
    )
