import pathlib

import numpy as np
import sklearn.externals.joblib

import pytoolkit as tk

base_dir = pathlib.Path(__file__).resolve().parent
data_dir = base_dir / 'data'


def _gen():
    gen = tk.image.ImageDataGenerator((64, 64), preprocess_input=lambda x: x)
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
    return gen


def test_image_data_generator_save():
    """画像の変換のテスト。目視したいので結果を`../___check/image[12]/`に保存しちゃう。"""
    gen = _gen()

    X_list = [
        np.array([sklearn.externals.joblib.load(str(data_dir / 'cifar.pkl'))]),
        np.array([str(data_dir / 'Lenna.png')]),
    ]
    for X, dir_name in zip(X_list, ['image1', 'image2']):
        save_dir = base_dir.parent / '___check' / dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        g = gen.flow(X, batch_size=1, data_augmentation=True, random_state=123)
        for i, X_batch in enumerate(g):
            assert X_batch.shape == (1, 64, 64, 3)
            tk.ndimage.save(save_dir / '{}.png'.format(i), X_batch[0])
            if i >= 31:
                break


def test_image_data_generator_repro():
    """再現性チェック。"""
    gen = _gen()

    X = np.array([
        str(data_dir / 'ai_pet_family.png'),
        str(data_dir / 'ai_shigoto_makaseru.png'),
        str(data_dir / 'ai_shigoto_ubau.png'),
    ])
    g1 = gen.flow(X, batch_size=2, data_augmentation=True, random_state=123)
    g2 = gen.flow(X, batch_size=2, data_augmentation=True, random_state=234)
    g3 = gen.flow(X, batch_size=2, data_augmentation=True, random_state=123)
    b1_1 = next(g1)
    b1_2 = next(g1)
    b2_1 = next(g2)
    b2_2 = next(g2)
    b3_1 = next(g3)
    b3_2 = next(g3)
    g1.close()
    g2.close()
    g3.close()
    assert b1_1.shape == (2, 64, 64, 3)
    assert b1_2.shape == (1, 64, 64, 3)
    assert b2_1.shape == (2, 64, 64, 3)
    assert b2_2.shape == (1, 64, 64, 3)
    assert b3_1.shape == (2, 64, 64, 3)
    assert b3_2.shape == (1, 64, 64, 3)
    assert (b1_1 != b2_1).any()
    assert (b1_2 != b2_2).any()
    assert (b1_1 == b3_1).all()
    assert (b1_2 == b3_2).all()
