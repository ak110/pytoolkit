#!/usr/bin/env python
"""ImageDataGeneratorのチェック用コード。"""
import pathlib

import better_exceptions
import numpy as np
from tqdm import tqdm

_BATCH_SIZE = 16
_ITER = 32
_IMAGE_SIZE = (256, 256)


def _main():
    better_exceptions.MAX_LENGTH = 128
    np.random.seed(1234)

    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'tests' / 'data'
    save_dir = base_dir / '___check' / 'bench'
    save_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(base_dir.parent))
    import pytoolkit as tk

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(_IMAGE_SIZE))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.25),
        tk.image.RandomBlur(probability=0.25, partial=True),
        tk.image.RandomUnsharpMask(probability=0.25),
        tk.image.RandomUnsharpMask(probability=0.25, partial=True),
        tk.image.RandomMedian(probability=0.25),
        tk.image.GaussianNoise(probability=0.25),
        tk.image.GaussianNoise(probability=0.25, partial=True),
        tk.image.RandomSaturation(probability=0.5),
        tk.image.RandomBrightness(probability=0.5),
        tk.image.RandomContrast(probability=0.5),
        tk.image.RandomHue(probability=0.5),
    ]))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(lambda x: x))
    gen.add(tk.image.ProcessOutput(lambda y: y))

    X = np.array([str(data_dir / '9ab919332a1dceff9a252b43c0fb34a0_m.jpg')] * 16)
    g = gen.flow(X, batch_size=_BATCH_SIZE, data_augmentation=True, random_state=123)
    # 適当にループして速度を見る
    with tqdm(total=_BATCH_SIZE * _ITER, unit='f', ascii=True, ncols=100) as pbar:
        for it, X_batch in enumerate(g):
            pbar.update(len(X_batch))
            if it + 1 >= _ITER:
                break
    # 最後のバッチを保存
    for ix, x in enumerate(X_batch):
        tk.ndimage.save(save_dir / '{}.png'.format(ix), x)


if __name__ == '__main__':
    _main()
