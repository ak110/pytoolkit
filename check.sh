#!/bin/bash
set -eux

black .

flake8

mypy pytoolkit scripts

pushd docs/
./update.sh
make html
popd

pylint -j0 pytoolkit scripts

CUDA_VISIBLE_DEVICES=none pytest

