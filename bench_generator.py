"""ImageDataGeneratorのチェック用コード。"""
import pathlib

import numpy as np
from tqdm import tqdm

_BATCH_SIZE = 16
_ITER = 32
_IMAGE_SIZE = (512, 512)


def _main():
    np.random.seed(1234)

    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir.joinpath('tests', 'data')
    save_dir = base_dir.joinpath('___check', 'bench')
    save_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(base_dir.parent))
    import pytoolkit as tk

    gen = tk.image.ImageDataGenerator(_IMAGE_SIZE)
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.5, tk.image.RandomErasing())
    gen.add(0.25, tk.image.RandomBlur())
    gen.add(0.25, tk.image.RandomBlur(partial=True))
    gen.add(0.25, tk.image.RandomUnsharpMask())
    gen.add(0.25, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.25, tk.image.RandomMedian())
    gen.add(0.25, tk.image.GaussianNoise())
    gen.add(0.25, tk.image.GaussianNoise(partial=True))
    gen.add(0.5, tk.image.RandomSaturation())
    gen.add(0.5, tk.image.RandomBrightness())
    gen.add(0.5, tk.image.RandomContrast())
    gen.add(0.5, tk.image.RandomHue())

    X = np.array([str(data_dir.joinpath('Lenna.png'))] * 16)
    g = gen.flow(X, batch_size=_BATCH_SIZE, data_augmentation=True, random_state=123)
    # 適当にループして速度を見る
    with tqdm(total=_BATCH_SIZE * _ITER, unit='f', ascii=True, ncols=100) as pbar:
        for it, X_batch in enumerate(g):
            pbar.update(len(X_batch))
            if it + 1 >= _ITER:
                # 最後のバッチを保存
                for ix, x in enumerate(X_batch):
                    tk.ndimage.save(save_dir.joinpath('{}.png'.format(ix)), x)
                break


if __name__ == '__main__':
    _main()
