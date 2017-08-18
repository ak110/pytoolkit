import pathlib

import numpy as np
import PIL.Image
import sklearn.externals.joblib

import pytoolkit as tk


def test_image_data_generator(tmpdir):
    data_dir = pathlib.Path(__file__).resolve().parent.joinpath('data')
    gen = tk.image.ImageDataGenerator((64, 64))

    X = np.array([sklearn.externals.joblib.load(str(data_dir.joinpath('test_image1.pkl')))])
    g = gen.flow(X, batch_size=1, data_augmentation=True, random_state=123)
    X_batch = g.__next__()
    assert X_batch.shape == (1, 64, 64, 3)
    img = PIL.Image.fromarray(((X_batch[0] / 2 + 0.5) * 255.).astype('uint8'), 'RGB')
    img.save(str(tmpdir.join('1.png')))
    g.close()

    X = np.array([str(data_dir.joinpath('test_image2.jpg'))])
    g = gen.flow(X, batch_size=1, data_augmentation=True, random_state=456)
    X_batch = g.__next__()
    assert X_batch.shape == (1, 64, 64, 3)
    img = PIL.Image.fromarray(((X_batch[0] / 2 + 0.5) * 255.).astype('uint8'), 'RGB')
    img.save(str(tmpdir.join('2.png')))
    g.close()
