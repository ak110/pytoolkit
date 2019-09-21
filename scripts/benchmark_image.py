#!/usr/bin/env python3
"""ImageDataGeneratorのチェック用コード。"""
import argparse
import cProfile
import pathlib
import sys

import albumentations as A
import numpy as np

if True:
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    import pytoolkit as tk

batch_size = 16
image_size = (512, 512)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    tk.utils.better_exceptions()
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    data_dir = base_dir / "tests" / "data"
    save_dir = base_dir / "___check" / "bench"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.load:
        X = np.array([data_dir / "9ab919332a1dceff9a252b43c0fb34a0_m.jpg"] * batch_size)
    else:
        X = tk.ndimage.load(data_dir / "9ab919332a1dceff9a252b43c0fb34a0_m.jpg")
        X = np.tile(np.expand_dims(X, axis=0), (batch_size, 1, 1, 1))
    if args.mask:
        y = X
    else:
        y = np.zeros((batch_size,))
    dataset = tk.data.Dataset(X, y)
    data_loader = MyDataLoader(
        data_augmentation=True, mask=args.mask, parallel=not args.profile
    )
    data_iterator = data_loader.iter(dataset, shuffle=True)

    if args.profile:
        # 適当にループして速度を見る
        cProfile.runctx(
            "_run(data_iterator, iterations=4)",
            globals=globals(),
            locals={"_run": _run, "data_iterator": data_iterator},
            sort="cumulative",
        )  # 累積:cumulative 内部:time
    else:
        # 1バッチ分を保存
        X_batch, _ = data_iterator[0]
        for ix, x in enumerate(X_batch):
            tk.ndimage.save(save_dir / f"{ix}.png", np.clip(x, 0, 255).astype(np.uint8))
        # 適当にループして速度を見る
        _run(data_iterator, iterations=16)

    print(f"{data_iterator.seconds_per_step * 1000:.0f}ms/step")


def _run(data_iterator, iterations):
    """ループして速度を見るための処理。"""
    with tk.utils.tqdm(total=batch_size * iterations, unit="f") as pbar:
        while True:
            for X_batch, y_batch in data_iterator:
                assert len(X_batch) == batch_size
                assert len(y_batch) == batch_size
                pbar.update(len(X_batch))
            data_iterator.on_epoch_end()
            if pbar.n >= pbar.total:
                break


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, data_augmentation, mask, parallel):
        super().__init__(batch_size=batch_size, parallel=parallel)
        self.data_augmentation = data_augmentation
        self.mask = mask
        if data_augmentation:
            self.aug = A.Compose(
                [
                    tk.image.RandomTransform(image_size[1], image_size[0]),
                    tk.image.RandomColorAugmentors(noisy=True),
                    tk.image.RandomErasing(),
                ]
            )
        else:
            self.aug = A.Compose([])

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_sample(index)
        X = tk.ndimage.load(X)
        if self.mask:
            y = tk.ndimage.load(y)
            a = self.aug(image=X, mask=y)
            X = a["image"]
            y = a["mask"]
        else:
            a = self.aug(image=X)
            X = a["image"]
        return X, y


if __name__ == "__main__":
    _main()
