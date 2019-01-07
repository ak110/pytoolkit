"""可視化関連。"""

import numpy as np


class GradCamVisualizer:
    """Grad-CAM(のようなもの)による可視化。

    Args:
        model: 対象のモデル。画像分類で最後がGlobalAveragePooling2D+Dense+softmaxで分類している前提。
        output_index: 使う出力(softmax)のインデックス。(クラスのindex)

    """

    def __init__(self, model, output_index):
        import keras
        # GlobalAveragePoolingへの入力テンソルを取得
        map_output = None
        for layer in model.layers[::-1]:
            if isinstance(layer, keras.layers.GlobalAveragePooling2D):
                map_output = layer.input
                break
        assert map_output is not None
        # 関数を作成
        grad = keras.backend.gradients(model.output[0, output_index], map_output)[0]
        mask = keras.backend.sum(map_output * grad, axis=-1)[0, :, :]
        mask_min, mask_max = keras.backend.min(mask), keras.backend.max(mask)
        mask = (mask - mask_min) / (mask_max - mask_min + keras.backend.epsilon())  # [0-1)
        self.get_mask_func = keras.backend.function(
            model.inputs + [keras.backend.learning_phase()], [mask])

    def draw(self, source_image, model_inputs, alpha=0.25, interpolation='nearest'):
        """ヒートマップ画像を作成して返す。

        Args:
            source_image: 元画像 (RGB。shape=(height, width, 3))
            model_inputs: モデルの入力1件分。(例えば普通の画像分類ならshape=(1, height, width, 3))
            alpha: ヒートマップの不透明度
            interpolation: マスクの拡大方法 (nearest, bilinear, bicubic, lanczos)

        Returns:
            画像 (RGB。shape=(height, width, 3))

        """
        import cv2
        assert source_image.shape[2:] == (3,)
        assert 0 < alpha < 1
        cv2_interp = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }[interpolation]

        mask = self.get_mask(model_inputs)
        mask = cv2.resize(mask, (source_image.shape[1], source_image.shape[0]), interpolation=cv2_interp)
        mask = np.uint8(256 * mask)  # [0-255]

        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = heatmap[..., ::-1]    # BGR to RGB

        result_image = np.uint8(heatmap * alpha + source_image * (1 - alpha))
        return result_image

    def get_mask(self, model_inputs):
        """可視化してマスクを返す。マスクの値は`[0, 1)`。"""
        if not isinstance(model_inputs, list):
            model_inputs = [model_inputs]
        mask = self.get_mask_func(model_inputs + [0])[0]
        assert len(mask.shape) == 2
        return mask
