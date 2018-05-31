#!/bin/bash
set -eux
GPU=$(nvidia-smi --list-gpus | wc -l)
mpirun -np $GPU python3 voc.py --network=experimental_large --input-size 640 640 --batch-size=8 $*
