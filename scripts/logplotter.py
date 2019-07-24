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

if True:
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    import pytoolkit as tk


def _main():
    tk.utils.better_exceptions()
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
        print(f"{args.logfile} items:")
        for col, _ in df_list:
            print(" ", col)
        print("")
    else:
        for col, df in df_list:
            if col == args.item:
                ax = df.plot(
                    xlim=(1, max(10, len(df))),
                    ylim=(
                        min(0, np.nanmin(df)) - 0.01,
                        max(
                            1,
                            min(
                                np.nanmax(df),
                                *(np.nanpercentile(df, [90, 99]) * [3, 2]),
                            ),
                        )
                        + 0.01,
                    ),
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
                    print(data_url)
                elif args.save:
                    save_path = pathlib.Path(f"{args.logfile.stem}.{args.item}.png")
                    save_path = save_path.resolve()
                    save_path.write_bytes(graph_bytes)
                    print(save_path)
                else:
                    data_url = tk.web.data_url(graph_bytes, "image/png")
                    b64data = base64.b64encode(data_url.encode("utf-8")).decode("utf-8")
                    print(f"\x1b]52;c;{b64data}\n\x1b\\")
                return
        raise RuntimeError(f'Item "{args.item}" is not found in {args.logfile}.')


def _parse_log(log_text: str):
    """Kerasのコンソール出力からlossなどを見つけてDataFrameに入れて返す。結果は(列名, DataFrame)の配列。"""
    pat1 = re.compile(r"Epoch +\d+: .+ time=\d+ ")
    pat2 = re.compile(r"\b(\w+)=([-+\.e\d]+|nan|-?inf)\b")
    data = {}
    for line in log_text.split("\n"):
        if not pat1.search(line):
            continue

        for m in pat2.finditer(line):
            key = m.group(1)
            value = float(m.group(2))
            if key not in data:
                data[key] = []
            data[key].append(value)

    if len(data) <= 0:
        return []

    max_length = max([len(v) for v in data.values()])
    for k, v in list(data.items()):
        if len(v) != max_length:
            data.pop(k)

    if len(data) == 0:
        return []

    df = pd.DataFrame.from_dict(data)
    df.index += 1

    df_list = []
    columns = df.columns.values
    for col in columns:
        if col.startswith("val_") and col[4:] in columns:
            continue
        # acc, val_accなどはまとめる。
        targets = [col]
        val_col = "val_" + col
        if val_col in columns:
            targets.append(val_col)
        df_list.append((col, df[targets]))

    return df_list


if __name__ == "__main__":
    _main()
