#!/bin/bash
set -eux
GPU=$(nvidia-smi --list-gpus | wc -l)
mpirun -np $GPU -- python cifar100.py
