#!/usr/bin/env python3
"""ImageDataGeneratorのチェック用コード。"""
import pathlib

import numpy as np

_BATCH_SIZE = 16
_ITER = 32
_IMAGE_SIZE = (256, 256)


def _main():
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'tests' / 'data'
    save_dir = base_dir / '___check' / 'bench'
    save_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(base_dir.parent))
    import pytoolkit as tk

    gen = tk.image.ImageDataGenerator(profile=True)
    gen.add(tk.generator.ProcessOutput(tk.ml.to_categorical(10), batch_axis=True))
    gen.add(tk.image.Resize(_IMAGE_SIZE))
    gen.add(tk.image.Mixup(probability=1, num_classes=10))
    gen.add(tk.image.ToGrayScale())
    gen.add(tk.image.SamplewiseStandardize())
    gen.add(tk.image.RandomBinarize())
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(_IMAGE_SIZE))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomFlipTB(probability=0.5))
    gen.add(tk.image.RandomRotate90(probability=1))
    gen.add(tk.image.RandomColorAugmentors(probability=0.5))
    gen.add(tk.image.RandomAlpha(probability=0.5))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.generator.ProcessInput(lambda x: x))

    X = np.array([data_dir / '9ab919332a1dceff9a252b43c0fb34a0_m.jpg'] * 16)
    y = np.zeros((len(X),), dtype=int)
    seq = gen.flow(X, y, batch_size=_BATCH_SIZE, data_augmentation=True, random_state=123)
    # 適当にループして速度を見る
    X_batch = []
    with tk.tqdm(total=_BATCH_SIZE * _ITER, unit='f') as pbar:
        for it, (X_batch, y_batch) in enumerate(seq):
            assert len(y_batch.shape) == 2
            pbar.update(len(X_batch))
            if it + 1 >= _ITER:
                break
    # 最後のバッチを保存
    for ix, x in enumerate(X_batch):
        tk.ndimage.save(save_dir / f'{ix}.png', x)

    # プロファイル結果
    gen.summary_profile()


if __name__ == '__main__':
    _main()
