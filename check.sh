#!/bin/bash
set -eux

pushd docs/
./update.sh
make html
popd

black --check .

flake8

pylint -j0 pytoolkit scripts

pytest

