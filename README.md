# pytoolkit

[![Build Status](https://travis-ci.org/ak110/pytoolkit.svg?branch=master)](https://travis-ci.org/ak110/pytoolkit)
[![Read the Docs](https://readthedocs.org/projects/ak110-pytoolkit/badge/?version=latest)](https://ak110-pytoolkit.readthedocs.io/ja/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

コンペなどで使いまわすコードを集めたもの。

いわゆるオレオレライブラリ。

`git submodule add https://github.com/ak110/pytoolkit.git` で配置して `import pytoolkit as tk` とかで使う。

## importするために最低限必要なライブラリ

- Pillow
- numpy
- numba
- opencv-python
- scikit-learn
- scipy
- tensorflow-gpu

## 使うときに動的にimportしている依存ライブラリ

- better-exceptions
- category_encoders
- chainercv
- diskcache
- h5py
- horovod
- matplotlib
- mpi4py
- onnxmltools
- optuna
- pandas
- pydensecrf
- tqdm
