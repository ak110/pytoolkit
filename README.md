# pytoolkit

[![Build Status](https://travis-ci.org/ak110/pytoolkit.svg?branch=master)](https://travis-ci.org/ak110/pytoolkit)

コンペなどで使いまわすコードを集めたもの。

いわゆるオレオレライブラリ。

`git submodule add https://github.com/ak110/pytoolkit.git` で配置して `import pytoolkit as tk` とかで使う。

## importするために最低限必要なライブラリ

- numpy
- scikit-learn
- scipy

## 使うときに動的にimportしている依存ライブラリ

- Pillow
- chainercv
- better_exceptions
- h5py
- horovod
- matplotlib
- mpi4py
- opencv-python
- pandas
- pydot
- sqlalchemy
- tensorflow-gpu
- tqdm
