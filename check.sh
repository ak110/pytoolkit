#!/bin/bash
set -eux

flake8

test "$(autopep8 --recursive --diff .)" = ""

pylint -j0 pytoolkit *.py

pytest

