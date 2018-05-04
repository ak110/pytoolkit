#!/bin/bash
set -eux
#
# README.md通りの環境でちゃんとimport出来るかチェック
#

SCRIPT="pip install numpy scikit-learn scipy ; python -c 'import pytoolkit ; print(\"OK!\")'"
docker run --interactive --tty --rm --volume="$PWD/..:/pytoolkit" python:3.6 bash -c "$SCRIPT"


