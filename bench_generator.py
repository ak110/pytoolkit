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

    gen = tk.image.ImageDataGenerator(profile=True)
    gen.add(tk.image.Resize(_IMAGE_SIZE))
    gen.add(tk.image.ProcessOutput(tk.ml.to_categorical(10), batch_axis=True))
    gen.add(tk.image.Mixup(probability=1))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(_IMAGE_SIZE))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.25),
        tk.image.RandomUnsharpMask(probability=0.25),
        tk.image.RandomMedian(probability=0.25),
        tk.image.GaussianNoise(probability=0.25),
        tk.image.RandomSaturation(probability=0.5),
        tk.image.RandomBrightness(probability=0.5),
        tk.image.RandomContrast(probability=0.5),
        tk.image.RandomHue(probability=0.5),
    ]))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(lambda x: x))

    X = np.array([str(data_dir / '9ab919332a1dceff9a252b43c0fb34a0_m.jpg')] * 16)
    y = np.zeros((len(X),), dtype=int)
    g = gen.flow(X, y, batch_size=_BATCH_SIZE, data_augmentation=True, random_state=123)
    # 適当にループして速度を見る
    X_batch = []
    with tqdm(total=_BATCH_SIZE * _ITER, unit='f', ascii=True, ncols=100) as pbar:
        for it, (X_batch, y_batch) in enumerate(g):
            assert len(y_batch.shape) == 2
            pbar.update(len(X_batch))
            if it + 1 >= _ITER:
                break
    # 最後のバッチを保存
    for ix, x in enumerate(X_batch):
        tk.ndimage.save(save_dir / '{}.png'.format(ix), x)

    # プロファイル結果
    gen.summary_profile()


if __name__ == '__main__':
    _main()
