#!/bin/bash
set -eux

black pytoolkit

flake8 pytoolkit

mypy pytoolkit

pushd docs/
./update.sh
make html
popd

pylint -j4 pytoolkit

CUDA_VISIBLE_DEVICES=none pytest pytoolkit
