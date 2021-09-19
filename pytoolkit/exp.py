"""実験結果の管理。"""
from __future__ import annotations

import datetime
import json
import pathlib
import shlex
import sys

import numpy as np

import pytoolkit as tk


class ExperimentLogger:
    """実験結果のログを管理するクラス。

    Args:
        output_dir: 出力先ディレクトリ
        name: 実験名とかメモ的なもの
        precision: ログ出力時の小数点以下の桁数

    """

    def __init__(
        self, output_dir: tk.typing.PathLike, name: str = None, precision: int = 3
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.name = name
        self.precision = precision

    def add(self, evals: tk.evaluations.EvalsType):
        """ログ出力。"""
        now = datetime.datetime.now()

        # experiments.json
        data_path = self.output_dir / "experiments.json"
        data = (
            json.loads(data_path.read_text(encoding="utf-8"))
            if data_path.exists()
            else {"entries": []}
        )
        data["entries"].append(
            {"time": now.isoformat(), "name": self.name, "evals": evals}
        )
        data_path.write_text(
            json.dumps(data, default=_default, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # report.txt
        commandline = " ".join([shlex.quote(a) for a in sys.argv])
        subject = f"[{now.isoformat()}] {commandline}"
        if self.name is not None:
            subject += f": {self.name}"
        body = tk.evaluations.to_str(evals, multiline=True, precision=self.precision)
        report_path = self.output_dir / "report.txt"
        report_path.write_text(f"{subject}\n{body}")

        # notifications
        tk.notifications.post(body=body, subject=subject)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.integer):
        return float(o)
    return repr(o)
