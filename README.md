# pytoolkit

[![Build Status](https://github.com/ak110/pytoolkit/actions/workflows/python-app.yml/badge.svg)](https://github.com/ak110/pytoolkit/actions/workflows/python-app.yml)
[![Read the Docs](https://readthedocs.org/projects/ak110-pytoolkit/badge/?version=latest)](https://ak110-pytoolkit.readthedocs.io/ja/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

コンペなどで使いまわすコードを集めたもの。

いわゆるオレオレライブラリ。

## インストール

`pip install https://github.com/ak110/pytoolkit.git` もしくは `git clone`して `pip install --user -e .[all]` など。

## 使い方

```
import pytoolkit
import pytoolkit.pipelines # optional
import pytoolkit.lgb # optional

pytoolkit.logs.init("path/to/logfile.log")
```
