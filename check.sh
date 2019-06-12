#!/bin/bash
set -eux

black --check .

flake8

pylint -j0 pytoolkit scripts

pytest

