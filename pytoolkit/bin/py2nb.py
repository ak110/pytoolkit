#!/usr/bin/env python3
"""*.pyファイルを[#%%]の行で区切って*.ipynbファイルに変換するスクリプト。

- https://qiita.com/KoheiKanagu/items/bd2560661f150cf532af
- https://gist.github.com/KoheiKanagu/183fdedc486ae5986900c3319bc421c6

"""
import argparse
import pathlib
import sys
import typing

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
            nb = parse(path, file.readlines())
            with save_path.open("w") as f:
                nbformat.write(nb, f)
                logger.info(f"Generated: {save_path}")


def parse(path: pathlib.Path, code: typing.List[str]) -> nbformat.v4:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = []
    cell_value = ""
    cell_is_markdown = False

    def add_cell(i):
        s = cell_value.strip()
        if s != "":
            if cell_is_markdown:
                nb["cells"].append(nbformat.v4.new_markdown_cell(s))
                logger.info(f"Markdown cell added. ({path}:{i + 1})")
            else:
                nb["cells"].append(nbformat.v4.new_code_cell(s))
                logger.info(f"Code cell added. ({path}:{i + 1})")

    for i, line in enumerate(code):
        s = line.strip()
        if s == "#%%":
            add_cell(i)
            cell_value = ""
            cell_is_markdown = False
        elif s == "#%% [markdown]":
            add_cell(i)
            cell_value = ""
            cell_is_markdown = True
        elif s.startswith("#%% "):
            # 独自拡張: 「#%% !pip install ...」みたいにすると!pip installのセルになる感じ
            add_cell(i)
            cell_value = s[4:]
            cell_is_markdown = False
        elif s == "# flake8: noqa: E265":
            logger.info(f"Ignored line: {s} ({path}:{i + 1})")
        else:
            if s.startswith("#%%"):
                logger.warning(f"Unknown marker: {s} ({path}:{i + 1})")
            cell_value += line

    add_cell(len(code))
    return nb


if __name__ == "__main__":
    main()
