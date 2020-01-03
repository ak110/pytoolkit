#!/usr/bin/env python3
"""*.pyファイルを[#%%]の行で区切って*.ipynbファイルに変換するスクリプト。

- https://qiita.com/KoheiKanagu/items/bd2560661f150cf532af
- https://gist.github.com/KoheiKanagu/183fdedc486ae5986900c3319bc421c6

"""
import argparse
import pathlib
import sys

import nbformat

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
        description="*.pyファイルを[#%%]の行で区切って*.ipynbファイルに変換するスクリプト。"
    )
    parser.add_argument("files", type=pathlib.Path, nargs="+", help="対象ファイル")
    args = parser.parse_args()
    for path in args.files:
        save_path = path.with_suffix(".ipynb")
        with path.open() as file:
            nb = parse(file.readlines())
            with save_path.open("w") as f:
                nbformat.write(nb, f)
                logger.info(f"Generated: {save_path}")


def parse(code: list) -> nbformat.v4:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = []
    cell_value = ""

    for line in code:
        if line == "# flake8: noqa: E265\n":
            logger.info(f"ignore line: [{line.strip()}]")
            continue

        if line.startswith("#%%"):
            if cell_value:
                nb["cells"].append(nbformat.v4.new_code_cell(cell_value))
            cell_value = ""

        cell_value += line

    if cell_value:
        nb["cells"].append(nbformat.v4.new_code_cell(cell_value))
    return nb


if __name__ == "__main__":
    main()
