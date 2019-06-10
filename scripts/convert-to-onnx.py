#!/usr/bin/env python3
"""*.h5 を読んで *.onnx で保存するスクリプト。"""
import argparse
import os
import pathlib
import sys

if True:
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    import pytoolkit as tk


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser(description='*.h5 を読んで *.onnx で保存するスクリプト。')
    parser.add_argument('model_path', type=pathlib.Path, help='対象ファイルのパス(*.h5)')
    args = parser.parse_args()

    with tk.dl.session():
        print(f'{args.model_path} Loading...')
        model = tk.models.load(args.model_path)

        save_path = args.model_path.with_suffix('.onnx')
        print(f'{save_path} Saving...')
        tk.models.save_onnx(model, save_path)
        print('Finished!')


if __name__ == '__main__':
    _main()
