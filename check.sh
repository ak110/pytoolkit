#!/bin/bash
set -eux

black .

flake8

mypy pytoolkit

pushd docs/
./update.sh
make html
popd

pylint -j0 pytoolkit

CUDA_VISIBLE_DEVICES=none pytest

