# pytoolkit

[![Build Status](https://travis-ci.org/ak110/pytoolkit.svg?branch=master)](https://travis-ci.org/ak110/pytoolkit)
[![Read the Docs](https://readthedocs.org/projects/ak110-pytoolkit/badge/?version=latest)](https://ak110-pytoolkit.readthedocs.io/ja/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

コンペなどで使いまわすコードを集めたもの。

いわゆるオレオレライブラリ。

`git submodule add https://github.com/ak110/pytoolkit.git pytoolkit.git && ln -s pytoolkit.git/pytoolkit` で配置して `import pytoolkit as tk` とかで使う。

(一応 `pip install --user -e .` とかもできるようにしているけどバージョニングとかはちゃんとしてないので基本的にはsubmoduleでコミット単位で紐付け。)

## cookiecutter

```bash
cookiecutter gh:ak110/cookiecutter-pytoolkit
```

<https://github.com/ak110/cookiecutter-pytoolkit>

## importするために最低限必要なライブラリ

- albumentations
  - Pillow
  - opencv-python-headless
  - scipy
  - numpy
- numba
- pandas
- scikit-learn
- tensorflow>=2.1.0

## 使うときに動的にimportしている依存ライブラリ

- better-exceptions
- catboost
- category_encoders
- chainercv
- h5py
- horovod
- ipython
- keras2onnx>=1.7.0
- lightgbm
- matplotlib
- mpi4py
- onnxmltools
- optuna
- python-dotenv
- requests
- tf2cv
- tf2onnx>=1.6.1
- tqdm
- xgboost
