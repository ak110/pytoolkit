"""CLI関連。"""
import argparse
import functools
import pathlib

import pytoolkit as tk


class App:
    """MLコンペとか用簡易フレームワーク。

    Args:
        output_dir (PathLike): ログ出力先ディレクトリ

    Fields:
        output_dir (pathlib.Path): ログ出力先ディレクトリ

    """

    def __init__(self, output_dir):
        self.output_dir = pathlib.Path(output_dir)
        self.inits = [tk.utils.better_exceptions]
        self.terms = []
        self.commands = {}

    def init(self):
        """前処理の追加用デコレーター"""

        def _decorator(func):
            self.inits.append(func)
            return func

        return _decorator

    def term(self):
        """後処理の追加用デコレーター"""

        def _decorator(func):
            self.terms.append(func)
            return func

        return _decorator

    def command(self):
        """コマンドの追加用デコレーター"""

        def _decorator(func):
            if func.__name__ in self.commands:
                raise ValueError(f"Duplicated command: {func.__name__}")
            _decorated_func = self._decoreate_command(func)
            self.commands[func.__name__] = _decorated_func
            return _decorated_func

        return _decorator

    def _decoreate_command(self, func):
        """コマンドに前処理などを付け加える。"""

        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            # ログ初期化
            tk.log.init(self.output_dir / f"{func.__name__}.log")
            # 前処理
            for f in self.inits:
                with tk.log.trace_scope(f.__qualname__):
                    f()
            try:
                with tk.log.trace_scope(func.__qualname__):
                    return func(*args, **kwargs)
            finally:
                # 後処理
                for f in self.terms:
                    with tk.log.trace_scope(f.__qualname__):
                        f()

        return _decorated_func

    def run(self, args=None, default=None):
        """実行。

        Args:
            args (list): 引数。(既定値はsys.argv)
            default (string): 未指定時に実行するコマンド名 (既定値は先頭のコマンド)

        """
        command_names = list(self.commands)
        default = default or command_names[0]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "command", choices=command_names, nargs="?", default=default
        )
        args = parser.parse_args(args)

        self.commands[args.command]()
