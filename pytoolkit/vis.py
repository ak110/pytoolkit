"""可視化関連。"""

import cv2
import numpy as np
import tensorflow as tf

K = tf.keras.backend


class GradCamVisualizer:
    """Grad-CAMによる可視化。

    Args:
        model: 対象のモデル。画像分類で最後がpooling_class+Dense+softmaxで分類している前提。
        output_index: 使う出力(softmax)のインデックス。(クラスのindex)
        pooling_class: 分類直前のpoolingのクラス。(既定値はGlobalAveragePooling2D)

    """

    def __init__(
        self,
        model: tf.keras.models.Model,
        output_index: int,
        pooling_class: type = None,
    ):
        pooling_class = pooling_class or tf.keras.layers.GlobalAveragePooling2D
        # pooling_classへの入力テンソルを取得
        map_output = None
        for layer in model.layers[::-1]:
            if isinstance(layer, pooling_class):
                map_output = layer.input
                break
        assert map_output is not None
        self.grad_model = tf.keras.models.Model(
            model.inputs, [map_output, model.output]
        )
        self.output_index = output_index

    def draw(
        self,
        source_image: np.ndarray,
        model_inputs: np.ndarray,
        alpha: float = 0.5,
        interpolation: str = "nearest",
    ) -> np.ndarray:
        """ヒートマップ画像を作成して返す。

        Args:
            source_image: 元画像 (RGB。shape=(height, width, 3))
            model_inputs: モデルの入力1件分。(例えば普通の画像分類ならshape=(1, height, width, 3))
            alpha: ヒートマップの不透明度
            interpolation: マスクの拡大方法 (nearest, bilinear, bicubic, lanczos)

        Returns:
            画像 (RGB。shape=(height, width, 3))

        """
        assert source_image.shape[2:] == (3,)
        assert 0 < alpha < 1
        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[interpolation]

        mask = self.get_mask(model_inputs)
        assert len(mask) == 1
        mask = mask[0, :, :]
        mask = cv2.resize(
            mask,
            (source_image.shape[1], source_image.shape[0]),
            interpolation=cv2_interp,
        )
        mask = (256 * mask).astype(np.uint8)  # [0-255]

        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_CIVIDIS)
        heatmap = heatmap[..., ::-1]  # BGR to RGB

        result_image = (heatmap * alpha + source_image * (1 - alpha)).astype(np.uint8)
        return result_image

    def get_mask(self, model_inputs):
        """可視化してマスクを返す。マスクの値は`[0, 1)`。"""
        mask = GradCamVisualizer._get_mask(
            self.grad_model, tf.constant(model_inputs), tf.constant(self.output_index)
        ).numpy()
        assert mask.ndim == 3  # (N, H, W)
        return mask

    @staticmethod
    @tf.function
    def _get_mask(grad_model, model_inputs, class_index):
        with tf.GradientTape() as tape:
            map_output, predictions = grad_model(model_inputs, training=False)
            model_output = predictions[:, class_index]
        grads = tape.gradient(model_output, map_output)
        mask = tf.keras.backend.sum(map_output * grads, axis=-1)
        mask = tf.nn.relu(mask) / (
            tf.keras.backend.max(mask) + tf.keras.backend.epsilon()
        )
        return mask
