# 実験用コード置き場

## 使い方例

    exp/check-import.sh

    PYTHONPATH=. exp/bench_generator.py

    PYTHONPATH=. mpirun -np $(nvidia-smi --list-gpus | wc -l) exp/cifar100.py


