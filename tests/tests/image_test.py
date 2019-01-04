import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import pytoolkit as tk


def _gen():
    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.Padding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize((64, 64)))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomFlipTB(probability=0.5))
    gen.add(tk.image.RandomRotate90(probability=1))
    gen.add(tk.image.RandomColorAugmentors())
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.generator.ProcessInput(lambda x: x))
    gen.add(tk.generator.ProcessOutput(lambda y: y))
    return gen


def test_image_data_generator_save(data_dir, check_dir):
    """画像の変換のテスト。目視したいので結果を`../___check/image[12]/`に保存しちゃう。"""
    gen = _gen()

    X_list = [
        np.array([joblib.load(str(data_dir / 'cifar.pkl'))]),
        np.array([str(data_dir / 'Lenna.png')]),
    ]
    for X, dir_name in zip(X_list, ['image1', 'image2']):
        save_dir = check_dir / dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        g, _ = gen.flow(X, batch_size=1, data_augmentation=True, random_state=123)
        for i, X_batch in enumerate(g):
            assert X_batch.shape == (1, 64, 64, 3)
            tk.ndimage.save(save_dir / f'{i}.png', X_batch[0])
            if i >= 31:
                break


def test_image_data_generator_repro(data_dir):
    """再現性チェック。"""
    gen = _gen()

    X = np.array([
        str(data_dir / 'ai_pet_family.png'),
        str(data_dir / 'ai_shigoto_makaseru.png'),
        str(data_dir / 'ai_shigoto_ubau.png'),
    ])
    g1, steps1 = gen.flow(X, batch_size=2, data_augmentation=True, random_state=123)
    g2, steps2 = gen.flow(X, batch_size=2, data_augmentation=True, random_state=234)
    g3, steps3 = gen.flow(X, batch_size=2, data_augmentation=True, random_state=123)
    assert steps1 == 2
    assert steps2 == 2
    assert steps3 == 2
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
