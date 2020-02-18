"""CLI関連。"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import typing

import tensorflow as tf

import pytoolkit as tk


class App:
    """MLコンペとか用簡易フレームワーク。

    ログの初期化とかのボイラープレートコードを出来るだけ排除するためのもの。

    Args:
        output_dir: ログ出力先ディレクトリ

    Attributes:
        output_dir: ログ出力先ディレクトリ
        current_command: 現在実行中のコマンド名

    """

    def __init__(self, output_dir: typing.Union[tk.typing.PathLike, None]):
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.inits: typing.List[typing.Callable[[], None]] = [
            tk.utils.better_exceptions,
            tk.math.set_ndarray_format,
        ]
        self.terms: typing.List[typing.Callable[[], None]] = []
        self.commands: typing.Dict[str, Command] = {}
        self.current_command: typing.Optional[str] = None

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

    def command(
        self,
        logfile: bool = True,
        then: str = None,
        use_horovod: bool = False,
        distribute_strategy_fn: typing.Callable[[], tf.distribute.Strategy] = None,
        args: typing.Dict[str, typing.Dict[str, typing.Any]] = None,
    ):
        """コマンドの追加用デコレーター。

        コマンド名は関数名。ただし_は-に置き換えたもの。

        Args:
            logfile: ログファイルを出力するのか否か
            then: 当該コマンドが終わった後に続けて実行するコマンドの名前
            use_horovod: horovodを使うならTrue
            distribute_strategy_fn: tf.distributeを使う場合のStrategyの作成関数
            args: add_argumentの引数。(例: args={"--x": {"type": int}})

        """
        assert not logfile or self.output_dir is not None

        def _decorator(entrypoint: typing.Callable):
            command_name = entrypoint.__name__.replace("_", "-")
            if command_name in self.commands:
                raise ValueError(f"Duplicated command: {command_name}")
            self.commands[command_name] = Command(
                entrypoint=entrypoint,
                logfile=logfile,
                then=then,
                use_horovod=use_horovod,
                distribute_strategy_fn=distribute_strategy_fn,
                args=args,
            )
            return entrypoint

        return _decorator

    def run(self, argv: typing.Sequence[str] = None, default: str = None):
        """実行。

        Args:
            argv: 引数。(既定値はsys.argv)
            default: 未指定時に実行するコマンド名 (既定値は先頭のコマンド)

        """
        commands = self.commands.copy()
        if "ipy" not in commands:
            commands["ipy"] = Command(entrypoint=_ipy, logfile=False)
        default = default or list(commands)[0]

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        for command_name, command in commands.items():
            p = subparsers.add_parser(command_name)
            for k, v in (command.args or {}).items():
                p.add_argument(k, **v)
        args = vars(parser.parse_args(argv))

        self.current_command = args["command"] or default
        args.pop("command")
        while True:
            assert self.current_command is not None
            command = commands[self.current_command]

            # horovod
            if command.use_horovod:
                tk.hvd.init()
            # ログ初期化
            tk.log.init(
                self.output_dir / f"{self.current_command}.log"
                if command.logfile and self.output_dir is not None
                else None
            )
            # 前処理
            for f in self.inits:
                with tk.log.trace_scope(f.__qualname__):
                    f()
            try:
                with tk.log.trace_scope(command.entrypoint.__qualname__):
                    if command.distribute_strategy_fn is not None:
                        command.distribute_strategy = command.distribute_strategy_fn()
                        with command.distribute_strategy.scope():
                            tk.log.get(__name__).info(
                                f"Number of devices: {self.num_replicas_in_sync}"
                            )
                            command.entrypoint(**args)
                    else:
                        command.entrypoint(**args)
            except Exception as e:
                # ログファイルを出力する(ような重要な)コマンドの場合のみ通知を送信
                if command.logfile:
                    tk.notifications.post(f"{type(e).__name__}: {e}")
                # ログ出力して強制終了 (これ以上raiseしても少しトレース増えるだけなので)
                tk.log.get(__name__).critical("Application error.", exc_info=True)
                sys.exit(1)
            finally:
                # 後処理
                for f in self.terms:
                    with tk.log.trace_scope(f.__qualname__):
                        f()

            # 次のコマンド
            self.current_command = command.then
            if self.current_command is None:
                break
            args = {}

    @property
    def num_replicas_in_sync(self) -> int:
        """現在のコマンドのdistribute_strategy.num_replicas_in_syncを取得する。"""
        if self.current_command is None:
            return 1
        command = self.commands.get(self.current_command)
        if command is None or command.distribute_strategy is None:
            return 1
        return command.distribute_strategy.num_replicas_in_sync


@dataclasses.dataclass()
class Command:
    """コマンド。

    Args:
        entrypoint: 呼び出される関数
        logfile: ログファイルを出力するのか否か
        then: 当該コマンドが終わった後に続けて実行するコマンドの名前
        use_horovod: horovodを使うならTrue
        distribute_strategy_fn: tf.distributeを使う場合のStrategyの作成関数
        distribute_strategy: 作成したStrategy
        args: add_argumentの引数。(例: args={"--x": {"type": int}})

    """

    entrypoint: typing.Callable
    logfile: bool = True
    then: typing.Optional[str] = None
    use_horovod: bool = False
    distribute_strategy_fn: typing.Optional[
        typing.Callable[[], tf.distribute.Strategy]
    ] = None
    distribute_strategy: typing.Optional[tf.distribute.Strategy] = None
    args: typing.Optional[typing.Dict[str, typing.Dict[str, typing.Any]]] = None


def _ipy():
    """自動追加されるコマンド。ipython。"""
    import IPython

    m = sys.modules["__main__"]
    user_ns = {k: getattr(m, k) for k in dir(m)}
    IPython.start_ipython(argv=["--ext=autoreload"], user_ns=user_ns)
