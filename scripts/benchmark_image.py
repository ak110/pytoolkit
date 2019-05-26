#!/usr/bin/env python3
"""ImageDataGeneratorのチェック用コード。"""
import argparse
import cProfile
import pathlib
import sys

import numpy as np

if True:
    sys.path.append('..')
    import pytoolkit as tk

_BATCH_SIZE = 16
_IMAGE_SIZE = (1024, 1024)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    tk.utils.better_exceptions()
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'tests' / 'data'
    save_dir = base_dir / '___check' / 'bench'
    save_dir.mkdir(parents=True, exist_ok=True)

    X = np.array([data_dir / '9ab919332a1dceff9a252b43c0fb34a0_m.jpg'] * _BATCH_SIZE)
    y = X
    dataset = MyDataset(X, y, None, data_augmentation=True)
    data_loader = tk.data.DataLoader(dataset, _BATCH_SIZE, shuffle=True, parallel=not args.profile)

    # 適当にループして速度を見る
    if args.profile:
        cProfile.runctx('_run(data_loader, iterations=4)',
                        globals=globals(),
                        locals={'_run': _run, 'data_loader': data_loader},
                        sort='cumulative')  # 累積:cumulative 内部:time
    else:
        _run(data_loader, iterations=16)

    print(f'{data_loader.seconds_per_step * 1000:.0f}ms/step')

    # 1バッチ分を保存
    for ix, x in enumerate(data_loader[0][0]):
        tk.ndimage.save(save_dir / f'{ix}.png', np.clip(x, 0, 255).astype(np.uint8))


def _run(data_loader, iterations):
    """ループして速度を見るための処理。"""
    with tk.utils.tqdm(total=_BATCH_SIZE * iterations, unit='f') as pbar:
        while True:
            for X_batch, y_batch in data_loader:
                assert len(X_batch) == _BATCH_SIZE
                assert len(y_batch) == _BATCH_SIZE
                pbar.update(len(X_batch))
            data_loader.on_epoch_end()
            if pbar.n >= pbar.total:
                break


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, num_classes, data_augmentation=False):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        if data_augmentation:
            self.aug = tk.image.Compose([
                tk.image.RandomTransform(_IMAGE_SIZE[1], _IMAGE_SIZE[0]),
                tk.image.RandomColorAugmentors(),
                tk.image.RandomErasing(),
            ])
        else:
            self.aug = tk.image.Compose([])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = tk.ndimage.load(self.X[index])
        y = tk.ndimage.load(self.y[index])
        a = self.aug(image=X, mask=y, rand=np.random.RandomState(index))
        X = a['image']
        y = a['mask']
        return X, y


if __name__ == '__main__':
    _main()
