"""DeepLearning(主にKeras)関連。"""
import os
import pathlib
import subprocess

import numpy as np


def get_gpu_count():
    """GPU数の取得。"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if gpus in ("-1", "none"):
            return 0
        return len(np.unique(gpus.split(",")))
    try:
        result_text = nvidia_smi("--list-gpus").strip()
        if "No devices found" in result_text:
            return 0
        return len([line for line in result_text.split("\n") if len(line) > 0])
    except FileNotFoundError:
        return 0


def nvidia_smi(*args):
    """nvidia-smiコマンドを実行する。"""
    path = (
        pathlib.Path(os.environ.get("ProgramFiles", ""))
        / "NVIDIA Corporation"
        / "NVSMI"
        / "nvidia-smi.exe"
    )
    if not path.is_file():
        path = pathlib.Path("nvidia-smi")
    command = [str(path)] + list(args)
    return subprocess.check_output(
        command, stderr=subprocess.STDOUT, universal_newlines=True
    )
