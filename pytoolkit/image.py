"""画像処理関連"""
import joblib  # pip install joblib
import numpy as np
import PIL
import PIL.Image
import PIL.ImageFilter
import sklearn.utils

from . import dl
from . import ndimage


class ImageDataGenerator(dl.Generator):
    """画像データのgenerator。

    Xは画像のファイルパスの配列またはndarray。
    ndarrayの場合は、(BGRではなく)RGB形式で、samples×rows×cols×channels。

    # 引数
    - image_size: 出力する画像のサイズのタプル(rows, cols)
    - grayscale: グレースケールで読み込むならTrue、RGBならFalse
    - preprocess_input: 前処理を行う関数。Noneなら`keras.application.inception_v3.preprocess_input`風処理。(画素値を-1～+1に線形変換)

    """

    def __init__(self, image_size=(300, 300), grayscale=False, preprocess_input=None,
                 padding_ratio=0.1,
                 padding_color=(0, 0, 0),
                 padding_on_resize=True,
                 crop_range=(0.9, 1.0),
                 aspect_ratio_list=(1, 3 / 4),
                 rotate_prob=0.125,
                 rotate_degree=5,
                 rotate90_prob=0,
                 blur_prob=0.125, blur_radius=1.0,
                 median_prob=0.125,
                 mirror_prob=0.5,
                 noise_prob=0.125, noise_scale=5,
                 saturation_prob=0.125, saturation_var=0.25,
                 brightness_prob=0.125, brightness_var=0.25,
                 contrast_prob=0.125, contrast_var=0.25,
                 lighting_prob=0.125, lighting_std=0.5):
        assert len(crop_range) == 2 and crop_range[0] <= crop_range[1]
        assert all([ar <= 1 for ar in aspect_ratio_list])
        # 出力の設定
        self.image_size = image_size
        self.grayscale = grayscale
        self.preprocess_input = preprocess_input
        # DataAugmentation関連
        self.padding_ratio = padding_ratio
        self.padding_color = padding_color
        self.padding_on_resize = padding_on_resize
        self.crop_range = crop_range
        self.aspect_ratio_list = aspect_ratio_list
        self.rotate_prob = rotate_prob
        self.rotate_degree = rotate_degree
        self.rotate90_prob = rotate90_prob
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius
        self.median_prob = median_prob
        self.mirror_prob = mirror_prob
        self.noise_prob = noise_prob
        self.noise_scale = noise_scale
        self.saturation_prob = saturation_prob
        self.saturation_var = saturation_var
        self.brightness_prob = brightness_prob
        self.brightness_var = brightness_var
        self.contrast_prob = contrast_prob
        self.contrast_var = contrast_var
        self.lighting_prob = lighting_prob
        self.lighting_std = lighting_std

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, random_state=None, data_augmentation=False):  # pylint: disable=arguments-differ
        """`fit_generator`などに渡すgenerator。

        # 引数
        - data_augmentation: Data Augmentationを行うか否か。
        """
        random_state = sklearn.utils.check_random_state(random_state)
        random_state2 = np.random.RandomState(random_state.randint(0, 2 ** 31))
        with joblib.Parallel(n_jobs=batch_size, backend='threading') as parallel:
            for tpl in super().flow(X, y, weights, batch_size, shuffle, random_state,
                                    parallel=parallel, data_augmentation=data_augmentation, rand=random_state2):
                yield tpl

    def _prepare(self, X, y=None, weights=None, parallel=None, data_augmentation=False, rand=None):  # pylint: disable=arguments-differ
        """画像の読み込みとDataAugmentation。"""
        seeds = rand.randint(0, 2 ** 31, len(X))
        jobs = [joblib.delayed(self._load)(x, data_augmentation, seed) for x, seed in zip(X, seeds)]
        X = np.array(parallel(jobs))
        return super()._prepare(X, y, weights)

    def _load(self, x, data_augmentation, seed):
        """画像の読み込みとDataAugmentation。(単数)"""
        rand = np.random.RandomState(seed)  # 再現性やスレッドセーフ性に自信がないのでここではこれを使う。

        # 画像の読み込み
        color_mode = 'L' if self.grayscale else 'RGB'
        img = ndimage.load_pillow(x, color_mode)

        # Data Augmentationその1 (PILでの処理)
        if data_augmentation:
            img = self._data_augmentation1(rand, img, color_mode)
        # リサイズ
        if self.image_size != img.size[::-1]:
            img = self._resize(img)
        # numpy配列化
        x = np.asarray(img, dtype=np.float32)
        if len(x.shape) == 3:
            pass
        elif len(x.shape) == 2:
            x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: ', x.shape)

        # Data Augmentationその2 (numpy配列化後の処理)
        if data_augmentation:
            x = self._data_augmentation2(rand, x)
        # elif mirror:
        #   x = x[:, ::-1, :]

        # preprocess_input
        pi = self.preprocess_input if self.preprocess_input else preprocess_input_abs1
        x = np.expand_dims(x, axis=0)
        x = pi(x)
        x = np.squeeze(x, axis=0)
        return x

    def _data_augmentation1(self, rand, img, color_mode):
        """DAその1 (numpy配列化前の処理)"""
        # ±90度回転
        if self.rotate90_prob:
            r = rand.rand()
            if r < self.rotate90_prob:
                img = img.transpose(PIL.Image.ROTATE_90)
            elif r < self.rotate90_prob * 2:
                img = img.transpose(PIL.Image.ROTATE_270)
        # 回転
        image_size = img.size
        if rand.rand() < self.rotate_prob:
            # アスペクト比が極端じゃない場合のみ回転可とする
            if img.height < img.width * 3 and img.width < img.height * 3:
                rotate_deg = (rand.rand() * 2 - 1) * self.rotate_degree
                img = img.convert('RGBA')
                img = img.rotate(rotate_deg, resample=PIL.Image.BILINEAR, expand=True)
                bg = PIL.Image.new('RGBA', img.size, color=self.padding_color + (255,))
                img = PIL.Image.composite(img, bg, img)
                img = img.convert(color_mode)
        # Padding
        if self.padding_ratio:
            pad_h = int(round(image_size[0] * self.padding_ratio))
            pad_v = int(round(image_size[1] * self.padding_ratio))
            bg = PIL.Image.new(color_mode, (img.width + pad_h, img.height + pad_v), color=self.padding_color)
            bg.paste(img, (pad_h // 2, pad_v // 2, pad_h // 2 + img.width, pad_v // 2 + img.height))
            img = bg

        # Scale Augmentation / Aspect Ratio Augmentation
        ar = rand.choice(self.aspect_ratio_list) if self.aspect_ratio_list else 1
        crop_rate = (self.crop_range[1] - self.crop_range[0]) * rand.rand() + self.crop_range[0]
        if rand.rand() < 0.5:
            crop_w = int(round(image_size[0] * crop_rate * ar))
            crop_h = int(round(image_size[1] * crop_rate))
        else:
            crop_w = int(round(image_size[0] * crop_rate))
            crop_h = int(round(image_size[1] * crop_rate * ar))
        crop_x = rand.randint(0, image_size[0] - crop_w + 1)
        crop_y = rand.randint(0, image_size[1] - crop_h + 1)
        img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        return img

    def _resize(self, img):
        """imgのサイズをself.image_sizeに合わせる。

        `padding_on_resize`がTrueならアスペクト比を維持するようにパディングする。(中央寄せ)
        Falseなら維持しないで無理やり全体にリサイズ。
        """
        if self.padding_on_resize:
            resize_rate_w = self.image_size[1] / img.width
            resize_rate_h = self.image_size[0] / img.height
            resize_rate = min(resize_rate_w, resize_rate_h)
            resized_size = (int(img.width * resize_rate), int(img.height * resize_rate))
        else:
            resized_size = (img.width, img.height)
        if self.image_size == resized_size[::-1]:
            img = img.resize((self.image_size[1], self.image_size[0]), resample=PIL.Image.LANCZOS)
        else:
            black = PIL.Image.new(img.mode, self.image_size)
            img = img.resize(resized_size, resample=PIL.Image.LANCZOS)
            paste_xy = (self.image_size[1] - img.width) // 2, (self.image_size[0] - img.height) // 2
            black.paste(img, paste_xy + (paste_xy[0] + img.width, paste_xy[1] + img.height))
            img = black
        return img

    def _data_augmentation2(self, rand, x):
        """DAその2 (numpy配列化後の処理)"""
        # 左右反転
        if self.mirror_prob and rand.rand() < self.mirror_prob:
            x = x[:, ::-1, :]
        # 色など
        jitters = []
        if rand.rand() < self.blur_prob:
            jitters.append(lambda x, rand: ndimage.blur(x, self.blur_radius * rand.rand()))
        if rand.rand() < self.median_prob:
            jitters.append(lambda x, rand: ndimage.median(x, size=2 if rand.rand() < 0.5 else 3))
        if rand.rand() < self.noise_prob:
            jitters.append(lambda x, rand: ndimage.gaussian_noise(x, rand, self.noise_scale))
        if rand.rand() < self.saturation_prob:
            jitters.append(lambda x, rand: ndimage.saturation(x, rand.uniform(1 - self.saturation_var, 1 + self.saturation_var)))
        if rand.rand() < self.brightness_prob:
            jitters.append(lambda x, rand: ndimage.brightness(x, rand.uniform(1 - self.brightness_var, 1 + self.brightness_var)))
        if rand.rand() < self.contrast_prob:
            jitters.append(lambda x, rand: ndimage.contrast(x, rand.uniform(1 - self.contrast_var, 1 + self.contrast_var)))
        if rand.rand() < self.lighting_prob:
            jitters.append(lambda x, rand: ndimage.lighting(x, rand.randn(3) * self.lighting_std))
        if jitters:
            rand.shuffle(jitters)
            for jitter in jitters:
                x = jitter(x, rand)
        # 色が範囲外になっていたら補正(飽和)
        x = np.clip(x, 0, 255)
        return x


def preprocess_input_mean(x: np.ndarray):
    """RGBそれぞれ平均値(定数)を引き算。

    `keras.applications.imagenet_utils.preprocess_input` のようなもの。(ただし `channels_last` 限定)
    `keras.applications`のVGG16/VGG19/ResNet50で使われる。
    """
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x


def preprocess_input_abs1(x: np.ndarray):
    """0～255を-1～1に変換。

    `keras.applications`のInceptionV3/Xceptionで使われる。
    """
    x /= 127.5
    x -= 1
    return x
