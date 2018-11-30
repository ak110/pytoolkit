"""主にnumpy配列(rows×cols×channels(RGB))の画像処理関連。

float32でRGBで0～255として扱う。
"""
import io
import pathlib
import typing

import numpy as np
import scipy.stats


def load(path_or_array: typing.Union[np.ndarray, io.IOBase, str, pathlib.Path], grayscale=False) -> np.ndarray:
    """画像の読み込み。

    やや余計なお世話だけど今後のためにfloat32に変換して返す。
    """
    assert path_or_array is not None

    if isinstance(path_or_array, np.ndarray):
        # ndarrayならそのまま画像扱い
        img = np.copy(path_or_array)  # 念のためコピー
    else:
        suffix = pathlib.Path(path_or_array).suffix.lower() if isinstance(path_or_array, (str, pathlib.Path)) else None
        if suffix == '.npy':
            # .npyなら読み込んでそのまま画像扱い
            img = np.load(str(path_or_array))
        else:
            # PILで読み込む
            import PIL.Image
            with PIL.Image.open(path_or_array) as pil_img:
                target_mode = 'L' if grayscale else 'RGB'
                if pil_img.mode != target_mode:
                    pil_img = pil_img.convert(target_mode)
                img = np.asarray(pil_img, dtype=np.float32)

    if img is None:
        raise ValueError(f'Image load failed: {path_or_array}')
    if grayscale:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
    else:
        if img.shape[-1] != 3:
            raise ValueError(f'Image load failed: shape={path_or_array.shape}')
    if len(img.shape) != 3:
        raise ValueError(f'Image load failed: shape={path_or_array}')

    return img.astype(np.float32)


def save(path: typing.Union[str, pathlib.Path], img: np.ndarray):
    """画像の保存。

    やや余計なお世話だけど0～255にクリッピング(飽和)してから保存。
    """
    assert len(img.shape) == 3 and img.shape[-1] in (1, 3, 4)
    import PIL.Image
    path = pathlib.Path(path)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[-1] == 1:
        pil_img = PIL.Image.fromarray(np.squeeze(img, axis=-1), 'L')
    elif img.shape[2] == 3:
        pil_img = PIL.Image.fromarray(img, 'RGB')
    elif img.shape[2] == 4:
        pil_img = PIL.Image.fromarray(img, 'RGBA')
    else:
        raise RuntimeError(f'Unknown format: shape={img.shape}')
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(path)


def rotate(rgb: np.ndarray, degrees: float, expand=True, interp='lanczos', border_mode='edge') -> np.ndarray:
    """回転。"""
    import cv2
    cv2_interp = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
    }[interp]
    cv2_border = {
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT,
        'wrap': cv2.BORDER_WRAP,
    }[border_mode]
    size = (rgb.shape[1], rgb.shape[0])
    center = (size[0] // 2, size[1] // 2)
    m = cv2.getRotationMatrix2D(center=center, angle=degrees, scale=1.0)
    if expand:
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        ex_w = int(np.ceil(size[1] * sin + size[0] * cos))
        ex_h = int(np.ceil(size[1] * cos + size[0] * sin))
        m[0, 2] += (ex_w / 2) - center[0]
        m[1, 2] += (ex_h / 2) - center[1]
        size = (ex_w, ex_h)
    rgb = cv2.warpAffine(rgb, m, size, flags=cv2_interp, borderMode=cv2_border)
    if len(rgb.shape) == 2:
        rgb = np.expand_dims(rgb, axis=-1)
    return rgb


def pad(rgb: np.ndarray, width: int, height: int, padding='edge') -> np.ndarray:
    """パディング。width/heightはpadding後のサイズ。(左右/上下均等、端数は右と下につける)"""
    assert width >= 0
    assert height >= 0
    x1 = max(0, (width - rgb.shape[1]) // 2)
    y1 = max(0, (height - rgb.shape[0]) // 2)
    x2 = width - rgb.shape[1] - x1
    y2 = height - rgb.shape[0] - y1
    rgb = pad_ltrb(rgb, x1, y1, x2, y2, padding)
    assert rgb.shape[1] == width and rgb.shape[0] == height
    return rgb


def pad_ltrb(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding='edge'):
    """パディング。x1/y1/x2/y2は左/上/右/下のパディング量。"""
    assert x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0
    assert padding in ('edge', 'zero', 'half', 'one', 'reflect', 'wrap', 'mean')
    kwargs = {}
    if padding == 'zero':
        mode = 'constant'
    elif padding == 'half':
        mode = 'constant'
        kwargs['constant_values'] = (127.5,)
    elif padding == 'one':
        mode = 'constant'
        kwargs['constant_values'] = (255.0,)
    elif padding == 'mean':
        mode = 'constant'
        kwargs['constant_values'] = (rgb.mean(),)
    else:
        mode = padding

    rgb = np.pad(rgb, ((y1, y2), (x1, x2), (0, 0)), mode=mode, **kwargs)
    return rgb


def crop(rgb: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """切り抜き。"""
    assert 0 <= x < rgb.shape[1]
    assert 0 <= y < rgb.shape[0]
    assert width >= 0
    assert height >= 0
    assert 0 <= x + width <= rgb.shape[1]
    assert 0 <= y + height <= rgb.shape[0]
    return rgb[y:y + height, x:x + width, :]


def flip_lr(rgb: np.ndarray) -> np.ndarray:
    """左右反転。"""
    return rgb[:, ::-1, :]


def flip_tb(rgb: np.ndarray) -> np.ndarray:
    """上下反転。"""
    return rgb[::-1, :, :]


def resize_long_side(rgb: np.ndarray, long_side: int, expand=True, interp='lanczos') -> np.ndarray:
    """長辺の長さを指定したアスペクト比維持のリサイズ。"""
    height, width = rgb.shape[:2]
    if not expand and max(width, height) <= long_side:
        return rgb
    if width >= height:  # 横長
        return resize(rgb, long_side, height * long_side // width, interp=interp)
    else:  # 縦長
        return resize(rgb, width * long_side // height, long_side, interp=interp)


def resize(rgb: np.ndarray, width: int, height: int, padding=None, interp='lanczos') -> np.ndarray:
    """リサイズ。"""
    import cv2
    assert interp in ('nearest', 'bilinear', 'bicubic', 'lanczos')
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    # パディングしつつリサイズ (縦横比維持)
    if padding is not None:
        resize_rate_w = width / rgb.shape[1]
        resize_rate_h = height / rgb.shape[0]
        resize_rate = min(resize_rate_w, resize_rate_h)
        resized_w = int(rgb.shape[0] * resize_rate)
        resized_h = int(rgb.shape[1] * resize_rate)
        rgb = resize(rgb, resized_w, resized_h, padding=None, interp=interp)
        return pad(rgb, width, height, padding=padding)
    # パディングせずリサイズ (縦横比無視)
    if rgb.shape[1] < width and rgb.shape[0] < height:  # 拡大
        cv2_interp = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }[interp]
    else:  # 縮小
        cv2_interp = cv2.INTER_NEAREST if interp == 'nearest' else cv2.INTER_AREA
    if rgb.shape[-1] in (1, 3, 4):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2_interp)
        if len(rgb.shape) == 2:
            rgb = np.expand_dims(rgb, axis=-1)
    else:
        resized_list = [cv2.resize(rgb[:, :, ch], (width, height), interpolation=cv2_interp) for ch in range(rgb.shape[-1])]
        rgb = np.transpose(resized_list, (1, 2, 0))
    return rgb


def gaussian_noise(rgb: np.ndarray, rand: np.random.RandomState, scale: float) -> np.ndarray:
    """ガウシアンノイズ。scaleは0～50くらい。小さいほうが色が壊れないかも。"""
    return rgb + rand.normal(0, scale, size=rgb.shape).astype(rgb.dtype)


def blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """ぼかし。sigmaは0～1程度がよい？"""
    import cv2
    rgb = cv2.GaussianBlur(rgb, (5, 5), sigma)
    if len(rgb.shape) == 2:
        rgb = np.expand_dims(rgb, axis=-1)
    return rgb


def unsharp_mask(rgb: np.ndarray, sigma: float, alpha=2.0) -> np.ndarray:
    """シャープ化。sigmaは0～1程度、alphaは1～2程度がよい？"""
    blured = blur(rgb, sigma)
    return rgb + (rgb - blured) * alpha


def median(rgb: np.ndarray, size: int) -> np.ndarray:
    """メディアンフィルタ。sizeは3程度がよい？"""
    import cv2
    rgb = cv2.medianBlur(rgb, size)
    if len(rgb.shape) == 2:
        rgb = np.expand_dims(rgb, axis=-1)
    return rgb


def brightness(rgb: np.ndarray, beta: float) -> np.ndarray:
    """明度の変更。betaの例：`np.random.uniform(-32, +32)`"""
    rgb += beta
    return rgb


def contrast(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """コントラストの変更。alphaの例：`np.random.uniform(0.75, 1.25)`"""
    rgb *= alpha
    return rgb


def saturation(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """彩度の変更。alphaの例：`np.random.uniform(0.5, 1.5)`"""
    gs = to_grayscale(rgb)
    rgb *= alpha
    rgb += (1 - alpha) * np.expand_dims(gs, axis=-1)
    return rgb


def hue_lite(rgb: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """色相の変更の適当バージョン。"""
    assert alpha.shape == (3,)
    assert beta.shape == (3,)
    assert (alpha > 0).all()
    rgb *= alpha / scipy.stats.hmean(alpha)
    rgb += beta - np.mean(beta)
    return rgb


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """グレースケール化。"""
    return rgb.dot([0.299, 0.587, 0.114])


def standardize(rgb: np.ndarray) -> np.ndarray:
    """標準化。0～255に適当に収める。"""
    rgb = (rgb - np.mean(rgb)) / (np.std(rgb) + 1e-5)
    rgb = rgb * 64 + 127
    return rgb.astype(np.float32)


def binarize(rgb: np.ndarray, threshold) -> np.ndarray:
    """二値化(白黒化)。"""
    assert 0 < threshold < 255
    return (rgb >= threshold).astype(np.float32) * 255


def rot90(rgb: np.ndarray, k) -> np.ndarray:
    """90度回転。"""
    assert 0 <= k <= 3
    if k == 1:
        rgb = np.swapaxes(rgb, 0, 1)[::-1, :, :]
    elif k == 2:
        rgb = rgb[::-1, ::-1, :]
    elif k == 3:
        rgb = np.swapaxes(rgb, 0, 1)[:, ::-1, :]
    return rgb


def equalize(rgb: np.ndarray) -> np.ndarray:
    """ヒストグラム平坦化。"""
    import cv2
    gray = np.clip(np.mean(rgb, axis=-1, keepdims=True), 0, 255)
    eq = np.expand_dims(cv2.equalizeHist(gray.astype(np.uint8)), axis=-1)
    rgb += eq - gray
    return rgb


def auto_contrast(rgb: np.ndarray, scale=255) -> np.ndarray:
    """オートコントラスト。"""
    gray = np.mean(rgb, axis=-1)
    b, w = gray.min(), gray.max()
    if b < w:
        rgb = (rgb - b) * (scale / (w - b))
    return rgb


def posterize(rgb: np.ndarray, bits) -> np.ndarray:
    """ポスタリゼーション。"""
    assert bits in range(1, 8 + 1)
    t = 2 ** (8 - bits) / 255
    return np.round(rgb * t) / t
