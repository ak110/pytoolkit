#!/usr/bin/env python3
"""tk.callbacks.EpochLogger()で出力したログからグラフを描画するスクリプト。"""
import argparse
import base64
import io
import pathlib
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pytoolkit as tk
except ImportError:
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
    import pytoolkit as tk

logger = tk.log.get(__name__)


def main():
    tk.utils.better_exceptions()
    tk.log.init(None)

    parser = argparse.ArgumentParser(
        description="tk.callbacks.EpochLogger()で出力したログからグラフを描画するスクリプト。"
    )
    parser.add_argument("logfile", type=pathlib.Path, help="対象のログファイルのパス")
    parser.add_argument("item", default=None, nargs="?", help="項目名。省略時は指定可能な項目名が表示される。")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--stdout", action="store_true", help="結果を標準出力に出力する。(既定ではOSC 52でクリップボード)"
    )
    g.add_argument("--save", action="store_true", help="結果をカレントディレクトリに画像ファイルとして出力する。")
    args = parser.parse_args()

    df_list = _parse_log(
        args.logfile.read_text(encoding="utf-8", errors="surrogateescape")
    )
    if args.item is None:
        logger.info(f"{args.logfile} items:")
        for col, _ in df_list:
            logger.info(f" {col}")
        logger.info("")
    else:
        for col, df in df_list:
            if col != args.item:
                continue
            xlim = (1, max(10, len(df)))
            yps = np.nanpercentile(df, [90, 99]) * [3, 2]
            ylim = (
                min(0, np.nanmin(df)) - 0.01,
                max(1, min(np.nanmax(df), *yps)) + 0.01,
            )
            ax = None
            for c in df.columns:
                ax = (
                    df[c]
                    .dropna()
                    .plot(
                        ax=ax,
                        xlim=xlim,
                        ylim=ylim,
                        marker="." if len(df) <= 1 or df[c].isnull().any() else None,
                        legend=True,
                    )
                )
            ax.set_xlabel("Epochs")
            ax.get_xaxis().set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True)
            )
            with io.BytesIO() as f:
                ax.get_figure().savefig(
                    f,
                    facecolor="w",
                    edgecolor="w",
                    orientation="portrait",
                    transparent=False,
                    format="PNG",
                )
                graph_bytes = f.getvalue()
            plt.close(ax.get_figure())

            if args.stdout:
                data_url = tk.web.data_url(graph_bytes, "image/png")
                logger.info(data_url)
            elif args.save:
                save_path = pathlib.Path(f"{args.logfile.stem}.{args.item}.png")
                save_path = save_path.resolve()
                save_path.write_bytes(graph_bytes)
                logger.info(save_path)
            else:
                data_url = tk.web.data_url(graph_bytes, "image/png")
                b64data = base64.b64encode(data_url.encode("utf-8")).decode("utf-8")
                print(f"\x1b]52;c;{b64data}\n\x1b\\")
            return
        raise RuntimeError(f'Item "{args.item}" is not found in {args.logfile}.')


def _parse_log(log_text: str):
    """tk.callbacks.EpochLoggerのログファイルからlossなどを見つけてDataFrameに入れて返す。結果は(列名, DataFrame)の配列。"""
    pat1 = re.compile(r"Epoch +\d+: .+ time=\d+ ")
    pat2 = re.compile(r"\b(\w+)=([-+\.e\d]+|nan|-?inf)\b")
    keys: list = []
    data_rows = []
    for line in log_text.split("\n"):
        if not pat1.search(line):
            continue

        data_row = {}
        for m in pat2.finditer(line):
            key = m.group(1)
            value = float(m.group(2))
            data_row[key] = value
            if key not in keys:
                keys.append(key)

        if len(data_row) > 0:
            data_rows.append(data_row)

    if len(data_rows) == 0:
        return []

    df = pd.DataFrame(index=range(1, len(data_rows) + 1))
    for key in keys:
        df[key] = [row.get(key, None) for row in data_rows]

    df_list = []
    for key in keys:
        if key.startswith("val_") and key[4:] in keys:
            continue
        # acc, val_accなどはまとめる。
        targets = [key]
        val_key = "val_" + key
        if val_key in keys:
            targets.append(val_key)
        df_list.append((key, df[targets]))

    return df_list


if __name__ == "__main__":
    main()
