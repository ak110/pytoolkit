"""主にnumpy配列(rows×cols×channels(RGB))の画像処理関連。

scipy.ndimageの薄いwrapperとか。
"""
import pathlib
from typing import Union

import numpy as np
import PIL
import PIL.Image
import PIL.ImageFilter
import scipy.misc
import scipy.ndimage


def load(path: Union[str, pathlib.Path], color_mode='RGB'):
    """画像の読み込み。

    やや余計なお世話だけど今後のためにfloat32に変換して返す。
    color_modeは'L'でグレースケール、'RGB'でRGB。
    """
    return scipy.misc.imread(str(path), mode=color_mode).astype(np.float32)


def save(path: Union[str, pathlib.Path], rgb: np.ndarray):
    """画像の保存。

    やや余計なお世話だけど0～255にクリッピング(飽和)してから保存。
    """
    rgb = np.clip(rgb, 0, 255)
    scipy.misc.imsave(str(path), rgb)


def load_pillow(x: Union[str, pathlib.Path, np.ndarray], color_mode='RGB') -> PIL.Image:
    """画像の読み込み。xはパスまたはndarray。"""
    if isinstance(x, (str, pathlib.Path)):
        img = PIL.Image.open(x)
        img = img.convert(color_mode)
    elif isinstance(x, np.ndarray):
        # 無駄だけどいったんPILに変換
        assert x.shape[-1] == (1 if color_mode == 'L' else 3)
        img = PIL.Image.fromarray(x, color_mode)
    else:
        raise ValueError('Invalid type: {}'.format(x))
    return img


def gaussian_noise(rgb: np.ndarray, rand: np.random.RandomState, scale: float):
    """ガウシアンノイズ。scaleは0～50くらい。小さいほうが色が壊れないかも。"""
    return rgb + rand.normal(0, scale, size=rgb.shape)

# jitters.append(lambda x, rand: scipy.ndimage.gaussian_filter(x, self.blur_radius * rand.rand()))
# jitters.append(lambda x, rand: scipy.ndimage.median_filter(x, size=2 if rand.rand() < 0.5 else 3))


def blur(rgb: np.ndarray, sigma: float):
    """ぼかし。sigmaは0～1程度がよい？"""
    return scipy.ndimage.gaussian_filter(rgb, [sigma, sigma, 0])


def unsharp_mask(rgb: np.ndarray, sigma: float, alpha=2.0):
    """シャープ化。sigmaは0～1程度、alphaは1～2程度がよい？"""
    blured = blur(rgb, sigma)
    return rgb + (rgb - blured) * alpha


def median(rgb: np.ndarray, size: int):
    """メディアンフィルタ。sizeは2 or 3程度がよい？"""
    channels = []
    for ch in range(rgb.shape[-1]):
        channels.append(scipy.ndimage.median_filter(rgb[:, :, ch], size))
    return np.stack(channels, axis=2)

# 以下は https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb から拝借。


def saturation(rgb: np.ndarray, alpha: float):
    """彩度の変更。alphaは(0.5～1.5)程度がよい。例：`np.random.uniform(0.5, 1.5)`"""
    gs = to_grayscale(rgb)
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return rgb


def brightness(rgb: np.ndarray, alpha: float):
    """明度の変更。alphaは(0.5～1.5)程度がよい。例：`np.random.uniform(0.5, 1.5)`"""
    rgb = rgb * alpha
    return rgb


def contrast(rgb: np.ndarray, alpha: float):
    """コントラストの変更。alphaは(0.5～1.5)程度がよい。例：`np.random.uniform(0.5, 1.5)`"""
    gs = to_grayscale(rgb).mean() * np.ones_like(rgb)
    rgb = rgb * alpha + (1 - alpha) * gs
    return rgb


def lighting(rgb: np.ndarray, rgb_noise: np.ndarray):
    """rgb_noiseは(-1～+1)程度の値の3要素の配列。例：`np.random.randn(3) * 0.5`"""
    cov = np.cov(rgb.reshape(-1, 3) / 255.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    rgb += eigvec.dot(eigval * rgb_noise) * 255
    return rgb


def to_grayscale(rgb: np.ndarray):
    """グレースケール化。"""
    return rgb.dot([0.299, 0.587, 0.114])
