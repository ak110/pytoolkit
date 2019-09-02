"""CLI関連。"""
import argparse
import pathlib

import pytoolkit as tk


class App:
    """MLコンペとか用簡易フレームワーク。

    ログの初期化とかのボイラープレートコードを出来るだけ排除するためのもの。

    Args:
        output_dir (PathLike): ログ出力先ディレクトリ

    Fields:
        output_dir (pathlib.Path): ログ出力先ディレクトリ
        current_command (str): 現在実行中のコマンド名

    """

    def __init__(self, output_dir):
        self.output_dir = pathlib.Path(output_dir)
        self.inits = [tk.utils.better_exceptions, tk.math.set_ndarray_format]
        self.terms = []
        self.commands = {}
        self.then = {}
        self.current_command = None

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

    def command(self, logfile=True, then=None):
        """コマンドの追加用デコレーター。

        Args:
            logfile (bool): ログファイルを出力するのか否か。
            then (str): 当該コマンドが終わった後に続けて実行するコマンドの名前。

        """

        def _decorator(func):
            if func.__name__ in self.commands:
                raise ValueError(f"Duplicated command: {func.__name__}")
            self.commands[func.__name__] = {
                "func": func,
                "logfile": logfile,
                "then": then,
            }
            return func

        return _decorator

    def run(self, args=None, default=None):
        """実行。

        Args:
            args (list): 引数。(既定値はsys.argv)
            default (string): 未指定時に実行するコマンド名 (既定値は先頭のコマンド)

        """
        commands = self.commands.copy()
        if "ipy" not in commands:
            commands["ipy"] = {"func": self._ipy, "logfile": False, "then": None}
        command_names = list(commands)
        default = default or command_names[0]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "command", choices=command_names, nargs="?", default=default
        )
        args = parser.parse_args(args)

        self.current_command = args.command
        while True:
            command = commands[self.current_command]

            # ログ初期化
            tk.log.init(
                self.output_dir / f"{command['func'].__name__}.log"
                if command["logfile"]
                else None
            )
            # 前処理
            for f in self.inits:
                with tk.log.trace_scope(f.__qualname__):
                    f()
            try:
                with tk.log.trace_scope(command["func"].__qualname__):
                    command["func"]()
            finally:
                # 後処理
                for f in self.terms:
                    with tk.log.trace_scope(f.__qualname__):
                        f()

            # 次のコマンド
            self.current_command = command["then"]
            if self.current_command is None:
                break

    def _ipy(self):
        """自動追加されるコマンド。ipython。"""
        import sys
        import IPython

        m = sys.modules["__main__"]
        user_ns = {k: getattr(m, k) for k in dir(m)}
        IPython.start_ipython(argv=[], user_ns=user_ns)
