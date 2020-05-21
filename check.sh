#!/bin/bash
set -eux

black .

flake8

mypy pytoolkit

pushd docs/
./update.sh
make html
popd

pylint -j4 pytoolkit

CUDA_VISIBLE_DEVICES=none pytest pytoolkit
