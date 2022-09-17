#!/usr/bin/env python3
"""*.h5 の値の最小・最大などを確認するスクリプト。"""
import argparse
import pathlib
import sys

import h5py
import numpy as np

try:
    import pytoolkit as tk
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    import pytoolkit as tk

logger = tk.log.get(__name__)


def main():
    tk.utils.better_exceptions()
    tk.log.init(None)
    parser = argparse.ArgumentParser(description="*.h5 の値の最小・最大などを確認するスクリプト。")
    parser.add_argument("model_path", type=pathlib.Path, help="対象ファイルのパス(*.h5)")
    args = parser.parse_args()

    logger.info(f"{args.model_path} Loading...")
    absmax_list = []
    with h5py.File(args.model_path, mode="r") as f:
        model_weights = f["model_weights"]
        layer_names = model_weights.attrs["layer_names"]
        for layer_name in layer_names:  # pylint: disable=not-an-iterable
            g = model_weights[layer_name]
            weight_names = g.attrs["weight_names"]
            for weight_name in weight_names:
                w = np.asarray(g[weight_name])
                key = f"/model_weights/{layer_name}/{weight_name}"
                if w.size == 1:
                    logger.info(f"{key}\t value={np.ravel(w)[0]:.2f}")
                else:
                    logger.info(
                        f"{key}\t min={w.min():.2f} max={w.max():.2f} mean={w.mean():.2f} std={w.std():.2f}"
                    )
                absmax_list.append((key, np.abs(w).max()))

    logger.info("abs Top-10:")
    for key, absvalue in list(sorted(absmax_list, key=lambda x: -x[1]))[:10]:
        logger.info(f"{absvalue:6.1f}: {key}")


if __name__ == "__main__":
    main()
