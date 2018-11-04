#!/bin/bash
set -eux
GPU=$(nvidia-smi --list-gpus | wc -l)
mpirun -np $GPU python3 od_voc.py train \
    --input-size 640 640 --batch-size=6 --result-dir=results_voc640 \
    --base-model=results_voc/model.h5 --epochs=100 \
    $*
python3 od_voc.py validate \
    --input-size 640 640 --batch-size=6 --result-dir=results_voc640 \
    $*
