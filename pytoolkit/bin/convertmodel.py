#!/usr/bin/env python3
"""*.h5 を読んでONNXなどに変換するスクリプト。"""
import argparse
import os
import pathlib
import sys

import tensorflow as tf

try:
    import pytoolkit as tk
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    import pytoolkit as tk

logger = tk.log.get(__name__)


def main():
    tk.utils.better_exceptions()
    tk.log.init(None)
    parser = argparse.ArgumentParser(
        description="hdf5/saved_model を読んでONNXなどに変換するスクリプト。"
    )
    parser.add_argument(
        "mode", choices=("hdf5", "saved_model", "onnx", "tflite"), help="変換先の形式"
    )
    parser.add_argument("model_path", type=pathlib.Path, help="対象ファイルのパス(*.h5)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "none"
    tf.keras.backend.set_learning_phase(0)

    tk.log.get(__name__).info(f"{args.model_path} Loading...")
    model = tk.models.load(args.model_path)

    if args.mode == "hdf5":
        save_path = args.model_path.with_suffix("h5")
    elif args.mode == "saved_model":
        save_path = args.model_path.with_suffix("")
    elif args.mode == "onnx":
        save_path = args.model_path.with_suffix(".onnx")
    elif args.mode == "tflite":
        save_path = args.model_path.with_suffix(".tflite")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    tk.log.get(__name__).info(f"{save_path} Saving...")
    tk.models.save(model, save_path, mode=args.mode)

    tk.log.get(__name__).info("Finished!")


if __name__ == "__main__":
    main()
