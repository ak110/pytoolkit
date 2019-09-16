"""学習結果の通知。

リポジトリにURLなどをコミットするのはいまいちなので、.envからURLを取得して通知。
無ければログ出力のみ。

- SLACK_URL: SlackのURL (https://hooks.slack.com/services/xxx/xxx/xxxxx)

"""
import json
import numbers
import pathlib
import os
import sys

import numpy as np

import pytoolkit as tk


def post_evals(evals):
    """学習結果を通知。

    Args:
        evals (dict): 評価結果などの入ったdict。

    """
    # 整形
    text = ""
    max_len = max([len(k) for k in evals])
    for k, v in evals.items():
        if isinstance(v, numbers.Number):
            text += f"{k}:{' ' * (max_len - len(k))} {v:.3f}\n"
        elif isinstance(v, np.ndarray):
            s = np.array_str(v, precision=3, suppress_small=True)
            text += f"{k}:{' ' * (max_len - len(k))} {s}\n"
        else:
            text += f"{k}:{' ' * (max_len - len(k))} {v}\n"
    # 通知
    post(text)


def post(text, username="<module_name>"):
    """通知。"""
    if tk.hvd.is_master():
        tk.log.get(__name__).debug(f"Notification:\n{text}")
        try:
            import dotenv
            import requests

            dotenv.load_dotenv()

            slack_url = os.environ.get("SLACK_URL", "")
            if len(slack_url) > 0:
                data = {"text": text}
                if username is not None:
                    if username == "<module_name>":
                        username = pathlib.Path(sys.argv[0]).name
                    data["username"] = username
                r = requests.post(
                    slack_url,
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"},
                )
                r.raise_for_status()
        except BaseException:
            tk.log.get(__name__).warning("Slackへの投稿失敗", exc_info=True)
    tk.hvd.barrier()
