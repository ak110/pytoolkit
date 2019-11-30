"""学習結果の通知。

環境変数や.envからURLを取得して通知する。(無ければログ出力のみ)

- SLACK_URL: SlackのIncoming WebhooksのURL <https://api.slack.com/incoming-webhooks>
  例: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX

- LINE_TOKEN: LINE Notifyのtoken <https://notify-bot.line.me/ja/>

"""
import json
import numbers
import os
import shlex
import sys

import numpy as np

import pytoolkit as tk


def post_evals(evals: dict):
    """学習結果を通知。

    Args:
        evals: 評価結果などの入ったdict。

    """
    # 整形
    text = ""
    max_len = max(len(k) for k in evals)
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


def post(text: str):
    """通知。"""
    pre_text = f"{' '.join([shlex.quote(a) for a in sys.argv])}\n"
    tk.log.get(__name__).debug(f"Notification:\n{text}")

    if tk.hvd.is_master():
        try:
            try:
                import dotenv

                dotenv.load_dotenv()
            except BaseException:
                tk.log.get(__name__).warning("dotenv読み込み失敗", exc_info=True)

            import requests

            slack_url = os.environ.get("SLACK_URL", "")
            if len(slack_url) > 0:
                data = {"text": f"{pre_text}```\n{text}```\n<!channel>"}
                r = requests.post(
                    slack_url,
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"},
                )
                r.raise_for_status()

            line_token = os.environ.get("LINE_TOKEN", "")
            if len(line_token) > 0:
                data = {"message": pre_text + text}
                r = requests.post(
                    "https://notify-api.line.me/api/notify",
                    data=data,
                    headers={"Authorization": "Bearer " + line_token},
                )
                r.raise_for_status()

        except BaseException:
            tk.log.get(__name__).warning("Slackへの投稿失敗", exc_info=True)
