#!/bin/bash
set -eux

black .

flake8

mypy pytoolkit

pushd tests/
./touch.sh
popd

pushd docs/
./update.sh
make html
popd

pylint -j0 pytoolkit scripts

pytest

