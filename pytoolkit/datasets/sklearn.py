"""sklearn.datasets関連。 <https://scikit-learn.org/stable/datasets/index.html>"""

import pytoolkit as tk


def load_boston():
    """<https://scikit-learn.org/stable/datasets/index.html#boston-dataset>"""
    import sklearn.datasets
    import pandas as pd

    bunch = sklearn.datasets.load_boston()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    return tk.data.Dataset(df, labels=bunch.target)


def load_lfw_pairs(*args, **kwargs):
    """<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html#sklearn.datasets.fetch_lfw_pairs>"""
    import sklearn.datasets

    bunch = sklearn.datasets.fetch_lfw_pairs(*args, **kwargs)
    return tk.data.Dataset(bunch.pairs, labels=bunch.target)
