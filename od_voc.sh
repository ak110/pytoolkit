#!/bin/bash
set -eux
GPU=$(nvidia-smi --list-gpus | wc -l)
mpirun -np $GPU python3 od_voc.py train $*
python3 od_voc.py validate $*
