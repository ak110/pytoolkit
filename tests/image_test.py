import pathlib

import numpy as np
import sklearn.externals.joblib

import pytoolkit as tk


def test_image_data_generator():
    """画像の変換のテスト。目視したいので結果を`../___check/image[12]/`に保存しちゃう。"""
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir.joinpath('data')
    save_dir = base_dir.parent.joinpath('___check', 'image1')
    save_dir.mkdir(parents=True, exist_ok=True)

    gen = tk.image.ImageDataGenerator((64, 64))
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.5, tk.image.RandomErasing())
    gen.add(0.125, tk.image.RandomBlur())
    gen.add(0.125, tk.image.RandomBlur(partial=True))
    gen.add(0.125, tk.image.RandomUnsharpMask())
    gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.125, tk.image.Sharp())
    gen.add(0.125, tk.image.Soft())
    gen.add(0.125, tk.image.RandomMedian())
    gen.add(0.125, tk.image.GaussianNoise())
    gen.add(0.125, tk.image.GaussianNoise(partial=True))
    gen.add(0.125, tk.image.RandomSaturation())
    gen.add(0.125, tk.image.RandomBrightness())
    gen.add(0.125, tk.image.RandomContrast())
    gen.add(0.125, tk.image.RandomHue())

    X = np.array([sklearn.externals.joblib.load(str(data_dir.joinpath('cifar.pkl')))])
    g = gen.flow(X, batch_size=1, data_augmentation=True, random_state=123)
    for i, X_batch in enumerate(g):
        assert X_batch.shape == (1, 64, 64, 3)
        tk.ndimage.save(save_dir.joinpath('{}.png'.format(i)), X_batch[0])
        if i >= 31:
            break

    save_dir = base_dir.parent.joinpath('___check', 'image2')
    save_dir.mkdir(parents=True, exist_ok=True)
    X = np.array([str(data_dir.joinpath('Lenna.png'))])
    g = gen.flow(X, batch_size=1, data_augmentation=True, random_state=456)
    for i, X_batch in enumerate(g):
        assert X_batch.shape == (1, 64, 64, 3)
        tk.ndimage.save(save_dir.joinpath('{}.png'.format(i)), X_batch[0])
        if i >= 31:
            break
