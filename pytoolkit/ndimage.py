"""主にnumpy配列(rows×cols×channels(RGB))の画像処理関連。

scipy.ndimageの薄いwrapperとか。
"""
import pathlib
from typing import Union

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal


def load(path: Union[str, pathlib.Path], color_mode='RGB') -> np.ndarray:
    """画像の読み込み。

    やや余計なお世話だけど今後のためにfloat32に変換して返す。
    color_modeは'L'でグレースケール、'RGB'でRGB。
    """
    return scipy.misc.imread(str(path), mode=color_mode).astype(np.float32)


def save(path: Union[str, pathlib.Path], rgb: np.ndarray) -> None:
    """画像の保存。

    やや余計なお世話だけど0～255にクリッピング(飽和)してから保存。
    """
    rgb = np.clip(rgb, 0, 255)
    scipy.misc.imsave(str(path), rgb)


def random_rotate(rgb: np.ndarray, rand: np.random.RandomState, degrees: float, padding='same') -> np.ndarray:
    """回転。"""
    return rotate(rgb, degrees=rand.uniform(-degrees, degrees), padding=padding)


def random_crop(rgb: np.ndarray, rand: np.random.RandomState,
                padding_rate=0.25, crop_rate=0.125,
                aspect_rations=(1, 1, 3 / 4, 4 / 3), padding='same') -> np.ndarray:
    """パディング＋ランダム切り抜き。"""
    cr = rand.uniform(1 - crop_rate, 1)
    ar = np.sqrt(rand.choice(aspect_rations))
    cropped_w = int(round(rgb.shape[1] * cr * ar))  # 元のサイズに対する割合
    cropped_h = int(round(rgb.shape[0] * cr / ar))
    padded_w = max(int(round(rgb.shape[1] * (1 + padding_rate))), cropped_w)
    padded_h = max(int(round(rgb.shape[0] * (1 + padding_rate))), cropped_h)
    # パディング
    rgb = pad(rgb, padded_w, padded_h, padding=padding)
    # 切り抜き
    x = rand.randint(0, rgb.shape[1] - cropped_w)
    y = rand.randint(0, rgb.shape[0] - cropped_h)
    return crop(rgb, x, y, cropped_w, cropped_h)


def rotate(rgb: np.ndarray, degrees: float, padding='same') -> np.ndarray:
    """回転。"""
    assert padding in ('same', 'zero')
    if padding == 'same':
        padding = 'nearest'
    elif padding == 'zero':
        padding = 'constant'
    return scipy.ndimage.rotate(rgb, degrees, reshape=True, mode=padding)


def pad(rgb: np.ndarray, width: int, height: int, padding='same') -> np.ndarray:
    """パディング。width/heightはpadding後のサイズ。(左右/上下均等、端数は右と下につける)"""
    assert width >= 0
    assert height >= 0
    assert padding in ('same', 'zero')
    x1 = max(0, (width - rgb.shape[1]) // 2)
    y1 = max(0, (height - rgb.shape[0]) // 2)
    x2 = width - rgb.shape[1] - x1
    y2 = height - rgb.shape[0] - y1
    rgb = pad_ltrb(rgb, x1, y1, x2, y2, padding)
    assert rgb.shape[1] == width and rgb.shape[0] == height
    return rgb


def pad_ltrb(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding='same'):
    """パディング。x1/y1/x2/y2は左/上/右/下のパディング量。"""
    assert padding in ('same', 'zero')
    if padding == 'same':
        padding = 'edge'
    elif padding == 'zero':
        padding = 'constant'
    return np.pad(rgb, ((y1, y2), (x1, x2), (0, 0)), mode=padding)


def crop(rgb: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """切り抜き。"""
    assert 0 <= x < rgb.shape[1]
    assert 0 <= y < rgb.shape[0]
    assert width >= 0
    assert height >= 0
    assert 0 <= x + width < rgb.shape[1]
    assert 0 <= y + height < rgb.shape[0]
    return rgb[y:y + height, x:x + width, :]


def flip_lr(rgb: np.ndarray) -> np.ndarray:
    """左右反転。"""
    return rgb[:, ::-1, :]


def flip_tb(rgb: np.ndarray) -> np.ndarray:
    """上下反転。"""
    return rgb[::-1, :, :]


def resize(rgb: np.ndarray, width: int, height: int, padding=None, interp='lanczos') -> np.ndarray:
    """リサイズ。"""
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    # パディングしつつリサイズ (縦横比維持)
    if padding is not None:
        assert padding in ('same', 'zero')
        resize_rate_w = width / rgb.shape[1]
        resize_rate_h = height / rgb.shape[0]
        resize_rate = min(resize_rate_w, resize_rate_h)
        resized_w = int(rgb.shape[0] * resize_rate)
        resized_h = int(rgb.shape[1] * resize_rate)
        if rgb.shape[1] != resized_w or rgb.shape[0] != resized_h:
            rgb = resize(rgb, resized_w, resized_h, padding=None, interp=interp)
        return pad(rgb, width, height)
    # パディングせずリサイズ (縦横比無視)
    if rgb.shape[-1] == 1:
        rgb = rgb.reshape(rgb.shape[:2])
    rgb = scipy.misc.imresize(rgb, (height, width), interp=interp).astype(np.float32)
    if len(rgb.shape) == 2:
        rgb = rgb.reshape(rgb.shape + (1,))
    return rgb


def gaussian_noise(rgb: np.ndarray, rand: np.random.RandomState, scale: float) -> np.ndarray:
    """ガウシアンノイズ。scaleは0～50くらい。小さいほうが色が壊れないかも。"""
    return rgb + rand.normal(0, scale, size=rgb.shape).astype(rgb.dtype)


def blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """ぼかし。sigmaは0～1程度がよい？"""
    return scipy.ndimage.gaussian_filter(rgb, [sigma, sigma, 0])


def unsharp_mask(rgb: np.ndarray, sigma: float, alpha=2.0) -> np.ndarray:
    """シャープ化。sigmaは0～1程度、alphaは1～2程度がよい？"""
    blured = blur(rgb, sigma)
    return rgb + (rgb - blured) * alpha


def sharp(rgb: np.ndarray) -> np.ndarray:
    """3x3のシャープ化。"""
    k = np.array([
        [+0.0, -0.2, +0.0],
        [-0.2, +1.8, -0.2],
        [+0.0, -0.2, +0.0],
    ], dtype=np.float32)
    channels = []
    for ch in range(rgb.shape[-1]):
        channels.append(scipy.signal.convolve2d(rgb[:, :, ch], k, mode='same', boundary='wrap'))
    return np.stack(channels, axis=2)


def soft(rgb: np.ndarray) -> np.ndarray:
    """3x3のぼかし。"""
    k = np.array([
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.2],
        [0.0, 0.2, 0.0],
    ], dtype=np.float32)
    channels = []
    for ch in range(rgb.shape[-1]):
        channels.append(scipy.signal.convolve2d(rgb[:, :, ch], k, mode='same', boundary='wrap'))
    return np.stack(channels, axis=2)


def median(rgb: np.ndarray, size: int) -> np.ndarray:
    """メディアンフィルタ。sizeは2 or 3程度がよい？"""
    channels = []
    for ch in range(rgb.shape[-1]):
        channels.append(scipy.ndimage.median_filter(rgb[:, :, ch], size))
    return np.stack(channels, axis=2)


def saturation(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """彩度の変更。alphaは(0.5～1.5)程度がよい。例：`np.random.uniform(0.5, 1.5)`"""
    gs = to_grayscale(rgb)
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return rgb


def brightness(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """明度の変更。alphaは(0.5～1.5)程度がよい。例：`np.random.uniform(0.5, 1.5)`"""
    rgb = rgb * alpha
    return rgb


def contrast(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """コントラストの変更。alphaは(0.5～1.5)程度がよい。例：`np.random.uniform(0.5, 1.5)`"""
    gs = to_grayscale(rgb).mean() * np.ones_like(rgb)
    rgb = rgb * alpha + (1 - alpha) * gs
    return rgb


def lighting(rgb: np.ndarray, rgb_noise: np.ndarray) -> np.ndarray:
    """rgb_noiseは(-1～+1)程度の値の3要素の配列。例：`np.random.randn(3) * 0.5`"""
    cov = np.cov(rgb.reshape(-1, 3) / 255.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    rgb += eigvec.dot(eigval * rgb_noise) * 255
    return rgb


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """グレースケール化。"""
    return rgb.dot([0.299, 0.587, 0.114]).astype(rgb.dtype)
