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
        name: 実験名。(過去との重複不可)

    """

    def __init__(self, output_dir: tk.typing.PathLike, name: str, precision: int = 3):
        self.output_dir = pathlib.Path(output_dir)
        self.name = name
        self.precision = precision
        # 名前の重複チェック
        data = self._load_data()
        if name in [e.get("name") for e in data["entries"]]:
            raise ValueError(f"Duplicate experiment name: {name}")

    def add(self, evals: tk.evaluations.EvalsType):
        """ログ出力。"""
        now = datetime.datetime.now()

        # experiments.json
        data = self._load_data()
        data["entries"].append(
            {"time": now.isoformat(), "name": self.name, "evals": evals}
        )
        self._save_data(data)

        # report.txt
        commandline = " ".join([shlex.quote(a) for a in sys.argv])
        subject = f"[{now.isoformat()}] {self.name} ({commandline})"
        body = tk.evaluations.to_str(evals, multiline=True, precision=self.precision)
        report_path = self.output_dir / "report.txt"
        report_path.write_text(f"{subject}\n{body}")

        # notifications
        tk.notifications.post(body=body, subject=subject)

    def _load_data(self):
        data_path = self.output_dir / "experiments.json"
        if not data_path.exists():
            return {"entries": []}
        return json.loads(data_path.read_text(encoding="utf-8"))

    def _save_data(self, data):
        data_path = self.output_dir / "experiments.json"

        def _default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            return repr(o)

        data_path.write_text(
            json.dumps(
                data,
                default=_default,
                ensure_ascii=False,
                indent=2,
                separators=(",", ": "),
            ),
            encoding="utf-8",
        )
