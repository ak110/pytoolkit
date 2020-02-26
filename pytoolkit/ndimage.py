"""主にnumpy配列(rows×cols×channels(RGB))の画像処理関連。

uint8のRGBで0～255として扱うのを前提とする。
あとグレースケールの場合もrows×cols×1の配列で扱う。
"""
import io
import pathlib
import random
import typing
import warnings

import cv2
import numba
import numpy as np
import PIL.Image
import PIL.ImageOps


def load(
    path_or_array: typing.Union[np.ndarray, io.IOBase, str, pathlib.Path],
    grayscale=False,
) -> np.ndarray:
    """画像の読み込みの実装。"""
    if isinstance(path_or_array, np.ndarray):
        # ndarrayならそのまま画像扱い
        img = np.copy(path_or_array)  # 念のためコピー
        assert img.dtype == np.uint8, f"ndarray dtype error: {img.dtype}"
    else:
        suffix = (
            pathlib.Path(path_or_array).suffix.lower()
            if isinstance(path_or_array, (str, pathlib.Path))
            else None
        )
        if suffix in (".npy", ".npz"):
            # .npyなら読み込んでそのまま画像扱い
            img = np.load(str(path_or_array))
            if isinstance(img, np.lib.npyio.NpzFile):
                if len(img.files) != 1:
                    raise ValueError(
                        f'Image load failed: "{path_or_array}" has multiple keys. ({img.files})'
                    )
                img = img[img.files[0]]
            assert img.dtype == np.uint8, f"{suffix} dtype error: {img.dtype}"
        else:
            # PILで読み込む
            try:
                with PIL.Image.open(path_or_array) as pil_img:
                    try:
                        pil_img = PIL.ImageOps.exif_transpose(pil_img)
                    except Exception as e:
                        warnings.warn(f"{type(e).__name__}: {e}")
                    try:
                        img = PIL.ImageOps.exif_transpose(img)
                    except Exception:
                        # Pillow 7.0.0で修正されるバグがある。
                        # https://github.com/python-pillow/Pillow/issues/3973
                        # これに限らず失敗時も害はそれほど無いと思われるので無視する。
                        pass
                    target_mode = "L" if grayscale else "RGB"
                    if pil_img.mode != target_mode:
                        pil_img = pil_img.convert(target_mode)
                    img = np.asarray(pil_img, dtype=np.uint8)
            except Exception as e:
                raise ValueError(f"Image load failed: {path_or_array}") from e

    if img is None:
        raise ValueError(f"Image load failed: {path_or_array}")
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    if len(img.shape) != 3:
        raise ValueError(f"Image load failed: shape={path_or_array}")

    return img


def get_image_size(
    path_or_array: typing.Union[np.ndarray, io.IOBase, str, pathlib.Path]
) -> typing.Tuple[int, int]:
    """画像サイズを取得する。(H, W)"""
    if isinstance(path_or_array, np.ndarray):
        # ndarrayならそのまま画像扱い
        img = path_or_array
        assert img.dtype == np.uint8, f"ndarray dtype error: {img.dtype}"
        return img.shape[:2]
    else:
        suffix = (
            pathlib.Path(path_or_array).suffix.lower()
            if isinstance(path_or_array, (str, pathlib.Path))
            else None
        )
        if suffix in (".npy", ".npz"):
            # .npyなら読み込んでそのまま画像扱い
            img = np.load(str(path_or_array))
            if isinstance(img, np.lib.npyio.NpzFile):
                if len(img.files) != 1:
                    raise ValueError(
                        f'Image load failed: "{path_or_array}" has multiple keys. ({img.files})'
                    )
                img = img[img.files[0]]
            assert img.dtype == np.uint8, f"{suffix} dtype error: {img.dtype}"
            return img.shape[:2]
        else:
            # PILで読み込む
            try:
                with PIL.Image.open(path_or_array) as pil_img:
                    try:
                        pil_img = PIL.ImageOps.exif_transpose(pil_img)
                    except Exception as e:
                        warnings.warn(f"{type(e).__name__}: {e}")
                    try:
                        pil_img = PIL.ImageOps.exif_transpose(pil_img)
                    except Exception:
                        # Pillow 7.0.0で修正されるバグがある。
                        # https://github.com/python-pillow/Pillow/issues/3973
                        # これに限らず失敗時も害はそれほど無いと思われるので無視する。
                        pass
                    return pil_img.height, pil_img.width
            except Exception as e:
                raise ValueError(f"Image load failed: {path_or_array}") from e


def save(
    path: typing.Union[str, pathlib.Path], img: np.ndarray, jpeg_quality: int = None
):
    """画像の保存。

    やや余計なお世話だけど0～255にクリッピング(飽和)してから保存する。

    Args:
        path: 保存先ファイルパス。途中のディレクトリが無ければ自動的に作成。
        img: 画像のndarray。shape=(height, width, 3)のRGB画像。dtypeはnp.uint8
        jpeg_quality: 1～100で指定する。

    """
    img = ensure_channel_dim(img)
    if img.dtype != np.uint8:
        warnings.warn(f"Invalid dtype: {img.dtype} (shape={img.shape})")

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = np.clip(img, 0, 255).astype(np.uint8)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(str(path), img)
        assert jpeg_quality is None
    elif suffix == ".npz":
        np.savez_compressed(str(path), img)
        assert jpeg_quality is None
    else:
        assert img.shape[-1] in (1, 3, 4)
        if img.shape[-1] == 1:
            pil_img = PIL.Image.fromarray(np.squeeze(img, axis=-1), "L")
        elif img.shape[2] == 3:
            pil_img = PIL.Image.fromarray(img, "RGB")
        elif img.shape[2] == 4:
            pil_img = PIL.Image.fromarray(img, "RGBA")
        else:
            raise RuntimeError(f"Unknown format: shape={img.shape}")
        kwargs = {}
        if jpeg_quality is not None:
            kwargs["quality"] = jpeg_quality
        pil_img.save(path, **kwargs)


def rotate(
    rgb: np.ndarray, degrees: float, expand=True, interp="lanczos", border_mode="edge"
) -> np.ndarray:
    """回転。"""
    cv2_interp = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[interp]
    cv2_border = {
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP,
    }[border_mode]
    m, w, h = compute_rotate(rgb.shape[1], rgb.shape[0], degrees=degrees, expand=expand)
    if rgb.shape[-1] in (1, 3):
        rgb = cv2.warpAffine(rgb, m, (w, h), flags=cv2_interp, borderMode=cv2_border)
        rgb = ensure_channel_dim(rgb)
    else:
        rotated_list = [
            cv2.warpAffine(
                rgb[:, :, ch], m, (w, h), flags=cv2_interp, borderMode=cv2_border
            )
            for ch in range(rgb.shape[-1])
        ]
        rgb = np.transpose(rotated_list, (1, 2, 0))
    return rgb


def compute_rotate(
    width: int, height: int, degrees: float, expand: bool = False
) -> tuple:
    """回転の変換行列を作成して返す。

    Args:
        width: 横幅
        height: 縦幅
        degrees: 回転する角度
        expand: Defaults to False. はみ出ないように画像を大きくするならTrue。

    Returns:
        変換行列、幅、高さ

    """
    center = (width // 2, height // 2)
    m = cv2.getRotationMatrix2D(center=center, angle=degrees, scale=1.0)
    if expand:
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        ex_w = int(np.ceil(height * sin + width * cos))
        ex_h = int(np.ceil(height * cos + width * sin))
        m[0, 2] += (ex_w / 2) - center[0]
        m[1, 2] += (ex_h / 2) - center[1]
        width, height = ex_w, ex_h
    return m, width, height


def pad(rgb: np.ndarray, width: int, height: int, padding="edge") -> np.ndarray:
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


def pad_ltrb(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding="edge"):
    """パディング。x1/y1/x2/y2は左/上/右/下のパディング量。"""
    assert x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0
    assert padding in ("edge", "zero", "half", "one", "reflect", "wrap", "mean")
    kwargs = {}
    if padding == "zero":
        mode = "constant"
    elif padding == "half":
        mode = "constant"
        kwargs["constant_values"] = (np.uint8(127),)
    elif padding == "one":
        mode = "constant"
        kwargs["constant_values"] = (np.uint8(255),)
    elif padding == "mean":
        mode = "constant"
        kwargs["constant_values"] = (np.uint8(rgb.mean()),)
    else:
        mode = padding

    rgb = np.pad(rgb, ((y1, y2), (x1, x2), (0, 0)), mode=mode, **kwargs)
    return rgb


@numba.njit(fastmath=True, nogil=True)
def crop(rgb: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """切り抜き。"""
    assert 0 <= x < rgb.shape[1]
    assert 0 <= y < rgb.shape[0]
    assert width >= 0
    assert height >= 0
    assert 0 <= x + width <= rgb.shape[1]
    assert 0 <= y + height <= rgb.shape[0]
    return rgb[y : y + height, x : x + width, :]


@numba.njit(fastmath=True, nogil=True)
def flip_lr(rgb: np.ndarray) -> np.ndarray:
    """左右反転。"""
    return rgb[:, ::-1, :]


@numba.njit(fastmath=True, nogil=True)
def flip_tb(rgb: np.ndarray) -> np.ndarray:
    """上下反転。"""
    return rgb[::-1, :, :]


def resize_long_side(
    rgb: np.ndarray, long_side: int, expand=True, interp="lanczos"
) -> np.ndarray:
    """長辺の長さを指定したアスペクト比維持のリサイズ。"""
    height, width = rgb.shape[:2]
    if not expand and max(width, height) <= long_side:
        return rgb
    if width >= height:  # 横長
        return resize(rgb, long_side, height * long_side // width, interp=interp)
    else:  # 縦長
        return resize(rgb, width * long_side // height, long_side, interp=interp)


def resize(
    rgb: np.ndarray, width: int, height: int, padding=None, interp="lanczos"
) -> np.ndarray:
    """リサイズ。"""
    assert interp in ("nearest", "bilinear", "bicubic", "lanczos")
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
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[interp]
    else:  # 縮小
        cv2_interp = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_AREA
    if rgb.ndim == 2 or rgb.shape[-1] in (1, 3):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2_interp)
        rgb = ensure_channel_dim(rgb)
    else:
        resized_list = [
            cv2.resize(rgb[:, :, ch], (width, height), interpolation=cv2_interp)
            for ch in range(rgb.shape[-1])
        ]
        rgb = np.transpose(resized_list, (1, 2, 0))
    return rgb


def gaussian_noise(
    rgb: np.ndarray, random_state: np.random.RandomState, scale: float
) -> np.ndarray:
    """ガウシアンノイズ。scaleは0～50くらい。小さいほうが色が壊れないかも。"""
    rgb = rgb + random_state.normal(0, scale, size=rgb.shape).astype(np.float32)
    return to_uint8(rgb)


def blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """ぼかし。sigmaは0～1程度がよい？"""
    rgb = cv2.GaussianBlur(rgb, (5, 5), sigma)
    rgb = ensure_channel_dim(rgb)
    return rgb


def unsharp_mask(rgb: np.ndarray, sigma: float, alpha=2.0) -> np.ndarray:
    """シャープ化。sigmaは0～1程度、alphaは1～2程度がよい？"""
    rgb = ensure_channel_dim(rgb)
    blured = blur(rgb, sigma)
    rgb = rgb.astype(np.float32)
    rgb = rgb + (rgb - blured) * alpha
    return to_uint8(rgb)


def median(rgb: np.ndarray, size: int) -> np.ndarray:
    """メディアンフィルタ。sizeは3程度がよい？"""
    rgb = cv2.medianBlur(rgb, size)
    rgb = ensure_channel_dim(rgb)
    return rgb


@numba.njit(fastmath=True, nogil=True)
def brightness(rgb: np.ndarray, beta: float) -> np.ndarray:
    """明度の変更。betaの例：np.random.uniform(-32, +32)"""
    return to_uint8(rgb.astype(np.float32) + np.float32(beta))


@numba.njit(fastmath=True, nogil=True)
def contrast(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """コントラストの変更。alphaの例：np.random.uniform(0.75, 1.25)"""
    # (rgb - 127.5) * alpha + 127.5
    # = rgb * alpha + 127.5 * (1 - alpha)
    alpha = np.float32(alpha)
    return to_uint8(rgb.astype(np.float32) * alpha + 127.5 * (1 - alpha))


@numba.njit(fastmath=True, nogil=True)
def saturation(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """彩度の変更。alphaの例：np.random.uniform(0.5, 1.5)"""
    rgb = rgb.astype(np.float32)
    gs = (rgb * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)).sum(axis=-1)
    return to_uint8(alpha * rgb + (1 - alpha) * np.expand_dims(gs, axis=-1))


@numba.njit(fastmath=True, nogil=True)
def hue_lite(rgb: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """色相の変更の適当バージョン。"""
    assert alpha.shape == (3,)
    assert beta.shape == (3,)
    assert (alpha > 0).all()
    ma = 3 / (1 / (alpha + 1e-7)).sum()
    mb = np.mean(beta.astype(np.float32))
    return to_uint8(rgb.astype(np.float32) * (alpha / ma) + (beta - mb))


@numba.njit(fastmath=True, nogil=True)
def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """グレースケール化。"""
    return to_uint8(
        (
            rgb.astype(np.float32)
            * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        ).sum(axis=-1)
    )


@numba.njit(fastmath=True, nogil=True)
def standardize(rgb: np.ndarray) -> np.ndarray:
    """標準化。0～255に適当に収める。"""
    rgb = rgb.astype(np.float32)
    rgb = (rgb - np.mean(rgb)) / (np.std(rgb) + 1e-5)
    rgb = rgb * 64 + 127
    return to_uint8(rgb)


@numba.njit(fastmath=True, nogil=True)
def binarize(rgb: np.ndarray, threshold) -> np.ndarray:
    """二値化(白黒化)。"""
    assert 0 < threshold < 255
    return np.where(rgb >= threshold, np.uint8(255), np.uint8(0))


@numba.njit(fastmath=True, nogil=True)
def rot90(rgb: np.ndarray, k) -> np.ndarray:
    """90度回転。"""
    assert 0 <= k <= 3
    if k == 1:
        rgb = rgb.transpose(1, 0, 2)[::-1, :, :]
    elif k == 2:
        rgb = rgb[::-1, ::-1, :]
    elif k == 3:
        rgb = rgb.transpose(1, 0, 2)[:, ::-1, :]
    return rgb


# @numba.njit(fastmath=True, nogil=True)
def equalize(rgb: np.ndarray) -> np.ndarray:
    """ヒストグラム平坦化。"""
    rgb = rgb.astype(np.float32)
    gray = np.mean(rgb, axis=-1, keepdims=True)
    eq = np.expand_dims(cv2.equalizeHist(gray.astype(np.uint8)), axis=-1)
    rgb = rgb + (eq - gray)
    return to_uint8(rgb)


# @numba.njit(fastmath=True, nogil=True)  # TypingError: numba doesn't support kwarg for mean
def auto_contrast(rgb: np.ndarray, scale=255) -> np.ndarray:
    """オートコントラスト。"""
    rgb = rgb.astype(np.float32)
    gray = np.mean(rgb, axis=-1)
    b, w = gray.min(), gray.max()
    if b < w:
        rgb = (rgb - b) * (scale / (w - b))
    return to_uint8(rgb)


# @numba.njit(fastmath=True, nogil=True)  # round
def posterize(rgb: np.ndarray, bits) -> np.ndarray:
    """ポスタリゼーション。"""
    assert bits in range(1, 8)
    t = np.float32(2 ** bits / 255)
    return to_uint8(np.round(rgb.astype(np.float32) * t) / t)


def geometric_transform(
    rgb: np.ndarray,
    output_width: int,
    output_height: int,
    flip_h: bool = False,
    flip_v: bool = False,
    degrees: float = 0,
    scale_h: float = 1.0,
    scale_v: float = 1.0,
    pos_h: float = 0.5,
    pos_v: float = 0.5,
    translate_h: float = 0.0,
    translate_v: float = 0.0,
    interp: str = "lanczos",
    border_mode: str = "edge",
) -> np.ndarray:
    """透視変換。

    Args:
        rgb: 入力画像
        output_width: 出力サイズ
        output_height: 出力サイズ
        flip_h: Defaults to False. Trueなら水平に反転する。
        flip_v: Defaults to False. Trueなら垂直に反転する。
        degrees: Defaults to 0. 回転する角度。(0や360なら回転無し。)
        scale_h: Defaults to 1.0. 水平方向のスケール。例えば0.5だと半分に縮小(zoom out / padding)、2.0だと倍に拡大(zoom in / crop)、1.0で等倍。
        scale_v: Defaults to 1.0. 垂直方向のスケール。例えば0.5だと半分に縮小(zoom out / padding)、2.0だと倍に拡大(zoom in / crop)、1.0で等倍。
        pos_h: Defaults to 0.5. スケール変換に伴う水平位置。0で左端、0.5で中央、1で右端。
        pos_v: Defaults to 0.5. スケール変換に伴う垂直位置。0で上端、0.5で中央、1で下端。
        translate_h: Defaults to 0.0. 変形元を水平にずらす量。-0.125なら12.5%左にずらし、+0.125なら12.5%右にずらす。
        translate_v: Defaults to 0.0. 変形元を垂直にずらす量。-0.125なら12.5%上にずらし、+0.125なら12.5%下にずらす。
        interp: Defaults to 'lanczos'. 補間方法。'nearest', 'bilinear', 'bicubic', 'lanczos'。縮小時は自動的にcv2.INTER_AREA。
        border_mode: Defaults to 'edge'. パディング方法。'edge', 'reflect', 'wrap'

    Returns:
        変換後画像

    """
    m = compute_perspective(
        rgb.shape[1],
        rgb.shape[0],
        output_width,
        output_height,
        flip_h=flip_h,
        flip_v=flip_v,
        degrees=degrees,
        scale_h=scale_h,
        scale_v=scale_v,
        pos_h=pos_h,
        pos_v=pos_v,
        translate_h=translate_h,
        translate_v=translate_v,
    )
    rgb = perspective_transform(
        rgb, output_width, output_height, m, interp=interp, border_mode=border_mode
    )
    return rgb


def compute_perspective(
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    flip_h: bool = False,
    flip_v: bool = False,
    degrees: float = 0,
    scale_h: float = 1.0,
    scale_v: float = 1.0,
    pos_h: float = 0.5,
    pos_v: float = 0.5,
    translate_h: float = 0.0,
    translate_v: float = 0.0,
) -> np.ndarray:
    """透視変換の変換行列を作成。

    Args:
        input_width: 入力サイズ
        input_height: 入力サイズ
        output_width: 出力サイズ
        output_height: 出力サイズ
        flip_h: Defaults to False. Trueなら水平に反転する。
        flip_v: Defaults to False. Trueなら垂直に反転する。
        degrees: Defaults to 0. 回転する角度。(0や360なら回転無し。)
        scale_h: Defaults to 1.0. 水平方向のスケール。例えば0.5だと半分に縮小(zoom out / padding)、2.0だと倍に拡大(zoom in / crop)、1.0で等倍。
        scale_v: Defaults to 1.0. 垂直方向のスケール。例えば0.5だと半分に縮小(zoom out / padding)、2.0だと倍に拡大(zoom in / crop)、1.0で等倍。
        pos_h: Defaults to 0.5. スケール変換に伴う水平位置。0で左端、0.5で中央、1で右端。
        pos_v: Defaults to 0.5. スケール変換に伴う垂直位置。0で上端、0.5で中央、1で下端。
        translate_h: Defaults to 0.0. 変形元を水平にずらす量。-0.125なら12.5%左にずらし、+0.125なら12.5%右にずらす。
        translate_v: Defaults to 0.0. 変形元を垂直にずらす量。-0.125なら12.5%上にずらし、+0.125なら12.5%下にずらす。

    Returns:
        変換行列

    """
    # 左上から時計回りに座標を用意
    src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    # 反転
    if flip_h:
        dst_points = dst_points[[1, 0, 3, 2]]
    if flip_v:
        dst_points = dst_points[[3, 2, 1, 0]]
    # 移動
    src_points[:, 0] -= translate_h
    src_points[:, 1] -= translate_v
    # 回転
    theta = degrees * np.pi * 2 / 360
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    src_points = np.dot(r, (src_points - 0.5).T).T + 0.5
    # スケール変換
    src_points[:, 0] /= scale_h
    src_points[:, 1] /= scale_v
    src_points[:, 0] -= (1 / scale_h - 1) * pos_h
    src_points[:, 1] -= (1 / scale_v - 1) * pos_v
    # 変換行列の作成
    src_points[:, 0] *= input_width
    src_points[:, 1] *= input_height
    dst_points[:, 0] *= output_width
    dst_points[:, 1] *= output_height
    m = cv2.getPerspectiveTransform(src_points, dst_points)
    return m


def perspective_transform(
    rgb: np.ndarray,
    width: int,
    height: int,
    m: np.ndarray,
    interp: str = "lanczos",
    border_mode: str = "edge",
) -> np.ndarray:
    """透視変換。

    Args:
        rgb: 入力画像
        width: 出力サイズ
        height: 出力サイズ
        m: 変換行列。
        interp: Defaults to 'lanczos'. 補間方法。'nearest', 'bilinear', 'bicubic', 'lanczos'。縮小時は自動的にcv2.INTER_AREA。
        border_mode: Defaults to 'edge'. パディング方法。'edge', 'reflect', 'wrap'

    Returns:
        変換後画像

    """
    cv2_interp = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[interp]
    cv2_border, borderValue = {
        "edge": (cv2.BORDER_REPLICATE, None),
        "reflect": (cv2.BORDER_REFLECT_101, None),
        "wrap": (cv2.BORDER_WRAP, None),
        "zero": (cv2.BORDER_CONSTANT, 0),
        "half": (cv2.BORDER_CONSTANT, 127),
        "one": (cv2.BORDER_CONSTANT, 255),
    }[border_mode]

    if cv2_interp != cv2.INTER_NEAREST:
        # 縮小ならINTER_AREA
        sh, sw = rgb.shape[:2]
        dr = transform_points([(0, 0), (sw, 0), (sw, sh), (0, sh)], m)
        dw = min(np.linalg.norm(dr[1] - dr[0]), np.linalg.norm(dr[2] - dr[3]))
        dh = min(np.linalg.norm(dr[3] - dr[0]), np.linalg.norm(dr[2] - dr[1]))
        if dw <= sw or dh <= sh:
            cv2_interp = cv2.INTER_AREA

    assert rgb.ndim in (2, 3)
    if rgb.ndim == 2 or rgb.shape[-1] in (1, 3):
        rgb = cv2.warpPerspective(
            rgb,
            m,
            (width, height),
            flags=cv2_interp,
            borderMode=cv2_border,
            borderValue=borderValue,
        )
        rgb = ensure_channel_dim(rgb)
    else:
        resized_list = [
            cv2.warpPerspective(
                rgb[:, :, ch],
                m,
                (width, height),
                flags=cv2_interp,
                borderMode=cv2_border,
                borderValue=borderValue,
            )
            for ch in range(rgb.shape[-1])
        ]
        rgb = np.transpose(resized_list, (1, 2, 0))
    assert rgb.ndim == 3
    return rgb


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """geometric_transformの座標変換。

    Args:
        points: 座標の配列。shape=(num_points, 2)。[(x, y)]
        m: 変換行列。

    Returns:
        変換後の座標の配列。

    """
    points = np.asarray(points)
    return cv2.perspectiveTransform(
        points.reshape((-1, 1, 2)).astype(np.float32), m
    ).reshape(points.shape)


def erase_random(
    rgb,
    random_state: np.random.RandomState,
    bboxes=None,
    scale_low=0.02,
    scale_high=0.4,
    rate_1=1 / 3,
    rate_2=3,
    alpha=None,
    max_tries=30,
):
    """Random erasing <https://arxiv.org/abs/1708.04896>"""
    if bboxes is not None:
        bb_lt = bboxes[:, :2]  # 左上
        bb_rb = bboxes[:, 2:]  # 右下
        bb_lb = bboxes[:, (0, 3)]  # 左下
        bb_rt = bboxes[:, (1, 2)]  # 右上
        bb_c = (bb_lt + bb_rb) / 2  # 中央

    for _ in range(max_tries):
        s = rgb.shape[0] * rgb.shape[1] * random_state.uniform(scale_low, scale_high)
        r = np.exp(random_state.uniform(np.log(rate_1), np.log(rate_2)))
        ew = int(np.sqrt(s / r))
        eh = int(np.sqrt(s * r))
        if ew <= 0 or eh <= 0 or ew >= rgb.shape[1] or eh >= rgb.shape[0]:
            continue
        ex = random_state.randint(0, rgb.shape[1] - ew)
        ey = random_state.randint(0, rgb.shape[0] - eh)

        if bboxes is not None:
            box_lt = np.array([[ex, ey]])
            box_rb = np.array([[ex + ew, ey + eh]])
            # bboxの頂点および中央を1つでも含んでいたらNGとする
            if (
                np.logical_and(box_lt <= bb_lt, bb_lt <= box_rb).all(axis=-1).any()
                or np.logical_and(box_lt <= bb_rb, bb_rb <= box_rb).all(axis=-1).any()
                or np.logical_and(box_lt <= bb_lb, bb_lb <= box_rb).all(axis=-1).any()
                or np.logical_and(box_lt <= bb_rt, bb_rt <= box_rb).all(axis=-1).any()
                or np.logical_and(box_lt <= bb_c, bb_c <= box_rb).all(axis=-1).any()
            ):
                continue
            # 面積チェック。塗りつぶされるのがbboxの面積の25%を超えていたらNGとする
            lt = np.maximum(bb_lt, box_lt)
            rb = np.minimum(bb_rb, box_rb)
            area_inter = np.prod(rb - lt, axis=-1) * (lt < rb).all(axis=-1)
            area_bb = np.prod(bb_rb - bb_lt, axis=-1)
            if (area_inter >= area_bb * 0.25).any():
                continue

        rgb = np.copy(rgb)
        rgb = ensure_channel_dim(rgb)
        rc = random_state.randint(0, 256, size=rgb.shape[-1])
        if alpha:
            rgb[ey : ey + eh, ex : ex + ew, :] = (
                rgb[ey : ey + eh, ex : ex + ew, :] * (1 - alpha) + rc * alpha
            ).astype(rgb.dtype)
        else:
            rgb[ey : ey + eh, ex : ex + ew, :] = rc[np.newaxis, np.newaxis, :].astype(
                rgb.dtype
            )
        break

    return rgb


def mixup(sample1: tuple, sample2: tuple, mode: str = "beta") -> tuple:
    """mixup。 <https://arxiv.org/abs/1710.09412>

    常に「sample1の重み >= sample2の重み」となるようにしている。

    Args:
        sample1: データその1
        sample2: データその2
        mode: 混ぜる割合の乱数の種類。
            - 'beta': β分布を0.5以上にした分布
            - 'uniform': [0.5, 1]の一様分布
            - 'uniform_ex': [0.5, √2]の一様分布

    Returns:
        tuple: 混ぜられたデータ。

    """
    if mode == "beta":
        r = np.float32(np.abs(random.betavariate(0.2, 0.2) - 0.5) + 0.5)
    elif mode == "uniform":
        r = np.float32(random.uniform(0.5, 1))
    elif mode == "uniform_ex":
        r = np.float32(random.uniform(0.5, np.sqrt(2)))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    assert r >= 0.5
    return mix_data(sample1, sample2, r)


@typing.overload
def mix_data(sample1: tuple, sample2: tuple, r: np.float32) -> tuple:
    # pylint: disable=function-redefined
    raise NotImplementedError()


@typing.overload
def mix_data(sample1: typing.Any, sample2: typing.Any, r: np.float32) -> typing.Any:
    # pylint: disable=function-redefined
    raise NotImplementedError()


def mix_data(sample1, sample2, r: np.float32):
    """mixup用に入力や出力を混ぜる処理。rはsample1側に掛ける率。"""
    # pylint: disable=function-redefined
    if sample1 is None:
        assert sample2 is None
        return None
    elif isinstance(sample1, tuple):
        assert isinstance(sample2, tuple)
        assert len(sample1) == len(sample2)
        return tuple(mix_data(s1, s2, r) for s1, s2 in zip(sample1, sample2))
    elif isinstance(sample1, list):
        assert isinstance(sample2, list)
        assert len(sample1) == len(sample2)
        return [mix_data(s1, s2, r) for s1, s2 in zip(sample1, sample2)]
    elif isinstance(sample1, dict):
        assert isinstance(sample2, dict)
        assert tuple(sample1.keys()) == tuple(sample2.keys())
        return {k: mix_data(sample1[k], sample2[k], r) for k in sample1}
    else:
        return np.float32(sample1) * r + np.float32(sample2) * (1 - r)


def cut_mix(sample1: tuple, sample2: tuple, beta: float = 1.0) -> tuple:
    """CutMix。 <https://arxiv.org/abs/1905.04899>

    Args:
        sample1: 画像とone-hot化したラベル(など)のタプル
        sample2: 画像とone-hot化したラベル(など)のタプル
        beta: beta分布のbeta

    Returns:
        image, label

    """
    image1, label1 = sample1
    image2, label2 = sample2
    assert image1.shape == image2.shape
    lam = random.betavariate(beta, beta)
    H, W = image1.shape[:2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = random.randrange(W)
    cy = random.randrange(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    image = np.copy(image1)  # 念のためコピー
    image[bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]
    label = mix_data(label1, label2, lam)
    return image, label


@numba.njit(fastmath=True, nogil=True)
def preprocess_tf(rgb):
    """RGB値の-1 ～ +1への変換"""
    return rgb.astype(np.float32) / np.float32(127.5) - 1


# @numba.njit(fastmath=True, nogil=True)
def mask_to_onehot(
    rgb: np.ndarray, class_colors: np.ndarray, append_bg: bool = False
) -> np.ndarray:
    """RGBのマスク画像をone-hot形式に変換する。

    Args:
        class_colors: 色の配列。shape=(num_classes, 3)
        append_bg: class_colorsに該当しない色のクラスを追加するならTrue。

    Returns:
        ndarray shape=(H, W, num_classes) dtype=np.float32

        append_bgがTrueの場合はnum_classesはlen(class_colors) + 1

    """
    num_classes = len(class_colors) + (1 if append_bg else 0)
    result = np.zeros((rgb.shape[0], rgb.shape[1], num_classes), np.float32)
    for i in range(len(class_colors)):
        result[np.all(rgb == class_colors[i], axis=-1), i] = 1
    if append_bg:
        result[:, :, -1] = 1 - result[:, :, :-1].sum(axis=-1)
    return result


# @numba.njit(fastmath=True, nogil=True)
def mask_to_class(
    rgb: np.ndarray, class_colors: np.ndarray, void_class: int = None
) -> np.ndarray:
    """RGBのマスク画像をクラスIDの配列に変換する。

    Args:
        class_colors: 色の配列。shape=(num_classes, 3)
        void_class: class_colorsに該当しない色のピクセルの値。Noneならlen(class_colors)

    Returns:
        ndarray shape=(H, W) dtype=np.int32

    """
    if void_class is None:
        void_class = len(class_colors)
    result = np.empty(rgb.shape[:2], dtype=np.int32)
    result[:] = void_class
    for i in range(len(class_colors)):
        result[np.all(rgb == class_colors[i], axis=-1)] = i
    return result


@numba.njit(fastmath=True, nogil=True)
def class_to_mask(classes: np.ndarray, class_colors: np.ndarray) -> np.ndarray:
    """クラスIDの配列をRGBのマスク画像に変換する。

    Args:
        classes: クラスIDの配列。 shape=(H, W)
        class_colors: 色の配列。shape=(num_classes, 3)

    Returns:
        ndarray shape=(H, W, 3)

    """
    return np.asarray(class_colors)[classes]


@numba.njit(fastmath=True, nogil=True)
def ensure_channel_dim(img: np.ndarray) -> np.ndarray:
    """shapeが(H, W)なら(H, W, 1)にして返す。それ以外ならそのまま返す。"""
    if img.ndim == 2:
        return np.expand_dims(img, axis=-1)
    assert img.ndim == 3, str(img.shape)
    return img


@numba.njit(fastmath=True, nogil=True)
def to_uint8(x):
    """floatからnp.uint8への変換。"""
    # np.clipは未実装: https://github.com/numba/numba/pull/3468
    return np.minimum(np.maximum(x, 0), 255).astype(np.uint8)


def perlin_noise(
    shape: typing.Tuple[int, int],
    frequency: float = 3.0,
    octaves: int = 5,
    ar: float = 1.0,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
):
    """パーリンノイズの生成。

    Args:
        shape: (H, W)
        frequency: 短辺の周波数
        octaves: 重ねる数
        ar: アスペクト比。>1なら横長、<1なら縦長のノイズを生成する

    """
    random_state = np.random.RandomState(seed=seed)
    rar = np.sqrt(ar)
    if shape[0] < shape[1]:
        freq_v = frequency / rar
        freq_h = frequency * shape[1] / shape[0] * rar
    else:
        freq_h = frequency * rar
        freq_v = frequency * shape[0] / shape[1] / rar
    grad_size_h = int(freq_h * lacunarity ** octaves + 1)
    grad_size_v = int(freq_v * lacunarity ** octaves + 1)
    gradient = random_state.uniform(-1, +1, size=(grad_size_h, grad_size_v, 2))

    image = np.zeros(shape, dtype=np.float32)
    amplify = 1.0
    for _ in range(octaves):
        image += _perlin_noise_base(shape, freq_v, freq_h, gradient) * amplify
        freq_v *= lacunarity
        freq_h *= lacunarity
        amplify *= persistence
    image -= image.min()
    image /= image.max()
    return np.uint8(image * 255)


def _perlin_noise_base(shape, freq_v, freq_h, gradient):
    # linear space by frequency
    x = np.tile(np.linspace(0, freq_h, shape[1], endpoint=False), shape[0])
    y = np.repeat(np.linspace(0, freq_v, shape[0], endpoint=False), shape[1])

    # gradient coordinates
    x0 = x.astype(int)
    y0 = y.astype(int)

    # local coordinate
    x -= x0
    y -= y0

    # gradient projections
    g00 = gradient[x0, y0]
    g10 = gradient[x0 + 1, y0]
    g01 = gradient[x0, y0 + 1]
    g11 = gradient[x0 + 1, y0 + 1]

    # fade
    t = (3 - 2 * x) * x * x

    # linear interpolation
    r = g00[:, 0] * x + g00[:, 1] * y
    s = g10[:, 0] * (x - 1) + g10[:, 1] * y
    g0 = r + t * (s - r)

    # linear interpolation
    r = g01[:, 0] * x + g01[:, 1] * (y - 1)
    s = g11[:, 0] * (x - 1) + g11[:, 1] * (y - 1)
    g1 = r + t * (s - r)

    # fade
    t = (3 - 2 * y) * y * y

    # (bi)linear interpolation
    g = g0 + t * (g1 - g0)

    # reshape
    return g.reshape(shape)
