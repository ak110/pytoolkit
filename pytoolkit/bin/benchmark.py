#!/usr/bin/env python3
"""ImageDataGeneratorの速度チェック用コード。

--profileを付けるとvmprofを使う。

sudo apt install libunwind-dev
pip install vmprof

OMP_NUM_THREADS=1 pytoolkit/bin/benchmark.py --profile
vmprofshow --prune_percent=5 benchmark.prof
rm benchmark.prof

"""
import argparse
import pathlib
import random
import sys
import time

import albumentations as A
import numpy as np

try:
    import pytoolkit as tk
except ImportError:
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
    import pytoolkit as tk

batch_size = 16
image_size = (512, 512)

logger = tk.log.get(__name__)


def main():
    tk.utils.better_exceptions()
    tk.log.init(None)

    random.seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "pytoolkit" / "_test_data"
    save_dir = base_dir / "___check" / "bench"
    save_dir.mkdir(parents=True, exist_ok=True)

    iterations = 128
    if args.load:
        X = np.array([data_dir / "9ab919332a1dceff9a252b43c0fb34a0_m.jpg"] * batch_size)
        iterations //= 2
    else:
        X = tk.ndimage.load(data_dir / "9ab919332a1dceff9a252b43c0fb34a0_m.jpg")
        X = np.tile(np.expand_dims(X, axis=0), (batch_size, 1, 1, 1))
    if args.mask:
        y = X
        iterations //= 2
    else:
        y = np.zeros((batch_size,))
    dataset = tk.data.Dataset(X, y)
    data_loader = MyDataLoader(data_augmentation=True, mask=args.mask)
    data_iterator = data_loader.iter(dataset, shuffle=True)

    if args.profile:
        import vmprof

        with pathlib.Path("benchmark.prof").open("w+b") as fd:
            vmprof.enable(fd.fileno())
            _run(data_iterator, iterations=iterations)
            vmprof.disable()
        logger.info("example: vmprofshow --prune_percent=5 benchmark.prof")
    else:
        # 1バッチ分を保存
        X_batch, _ = next(iter(data_iterator.ds))
        for ix, x in enumerate(X_batch):
            tk.ndimage.save(save_dir / f"{ix}.png", np.clip(x, 0, 255).astype(np.uint8))
        # 適当にループして速度を見る
        _run(data_iterator, iterations=iterations)


def _run(data_iterator, iterations):
    """ループして速度を見るための処理。"""
    start_time = time.perf_counter()
    with tk.utils.tqdm(total=batch_size * iterations, unit="f") as pbar:
        for X_batch, y_batch in data_iterator.ds:
            assert len(X_batch) == batch_size
            assert len(y_batch) == batch_size
            pbar.update(len(X_batch))
            if pbar.n >= pbar.total:
                break
    sps = (time.perf_counter() - start_time) / iterations
    logger.info(f"{sps * 1000:.0f}ms/step")


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, data_augmentation, mask):
        super().__init__(batch_size=batch_size)
        self.data_augmentation = data_augmentation
        self.mask = mask
        if data_augmentation:
            self.aug = A.Compose(
                [
                    tk.image.RandomTransform(size=image_size[:2]),
                    tk.image.RandomColorAugmentors(noisy=True),
                    tk.image.RandomErasing(),
                ]
            )
        else:
            self.aug = A.Compose([])

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
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
    main()
