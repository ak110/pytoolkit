"""sklearn.datasets関連。 <https://scikit-learn.org/stable/datasets/index.html>"""
import atexit
import pathlib
import tempfile

import numpy as np
import sklearn.datasets

import pytoolkit as tk


def load_breast_cancer() -> tk.data.Dataset:
    """<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer .html>"""
    # pylint: disable=no-member  # sklearn.utils.Bunchがうまく認識されないため
    bunch = sklearn.datasets.load_breast_cancer(as_frame=True)
    return tk.data.Dataset(bunch.data, labels=bunch.target)


def load_iris() -> tk.data.Dataset:
    """<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html>"""
    # pylint: disable=no-member  # sklearn.utils.Bunchがうまく認識されないため
    bunch = sklearn.datasets.load_iris(as_frame=True)
    return tk.data.Dataset(bunch.data, labels=bunch.target)


def fetch_california_housing() -> tk.data.Dataset:
    """<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>"""
    # pylint: disable=no-member  # sklearn.utils.Bunchがうまく認識されないため
    bunch = sklearn.datasets.fetch_california_housing(as_frame=True)
    return tk.data.Dataset(bunch.data, labels=bunch.target)


def load_lfw_pairs(**kwargs) -> tk.data.Dataset:
    """<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html>"""
    # pylint: disable=no-member  # sklearn.utils.Bunchがうまく認識されないため
    bunch = sklearn.datasets.fetch_lfw_pairs(**kwargs)
    return tk.data.Dataset(bunch.pairs, labels=bunch.target)


def load_sample_images() -> tk.data.Dataset:
    """テストコードなど用のサンプル画像読み込み。

    <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html>

    """
    # pylint: disable=no-member  # sklearn.utils.Bunchがうまく認識されないため
    bunch = sklearn.datasets.load_sample_images()

    # pylint: disable=consider-using-with
    temp_dir = tempfile.TemporaryDirectory()
    atexit.register(temp_dir.cleanup)

    save_dir = pathlib.Path(temp_dir.name)
    filenames = []
    for img, name in zip(bunch.images, bunch.filenames):
        save_path = save_dir / name
        tk.ndimage.save(save_path, img)
        filenames.append(save_path)
    return tk.data.Dataset(np.array(filenames), metadata={"temp_dir": temp_dir})
