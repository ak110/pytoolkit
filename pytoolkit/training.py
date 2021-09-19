"""学習関連。

tk.evaluations.EvalsType を返す関数を前提にしたヘルパー関数など。

"""
from __future__ import annotations

import logging
import typing

import pytoolkit as tk

logger = logging.getLogger(__name__)


def multi_run(
    func: typing.Callable[[], tk.evaluations.EvalsType], runs: int = 5
) -> tk.evaluations.EvalsType:
    """funcをruns回実行して結果の平均を返す。"""
    evals_list = []
    for i in range(runs):
        logger.info(f"multi run: {i + 1}/{runs}")
        evals_list.append(func())
    return tk.evaluations.mean(evals_list)


def hpo(
    func: typing.Callable[..., tk.evaluations.EvalsType],
    params: dict[str, tuple[str, dict[str, typing.Any]]],
    score_name: str,
    direction: str = "minimize",
    n_trials: int = 100,
) -> None:
    """ハイパーパラメータ探索。

    Args:
        params: キーがパラメータ名、値がtupleのdict。
        score_name: decorateした関数が返したdictのうちスコアとして使用する値を示すキー。
        direction: 最適化方向。minimize or maximize。

    paramsの値は、以下の2つの値のタプル
    - trial.suggest_*の「*」の部分 (str)
    - trial.suggest_*に渡す**kwargs (dict)

    例::

        @tk.training.hpo(
            params={
                "a": ("categorical", {"choices": [32, 64]}),
                "b": ("discrete_uniform", {"low": 0.1, "high": 1.0, "q": 0.1}),
                "c": ("float", {"low": 0.1, "high": 1.0, "step": 0.1, "log": False})
                "d": ("int", {"low": 1, "high": 10, "step": 1, "log": False})
                "e": ("loguniform", {"low": 1, "high": 10})
                "f": ("uniform", {"low": 1, "high": 10})
            },
            score_name="acc",
            direction="maximize",
            n_trials=100,
        )
        def train(a, b, c, d, e, f):
            return {"acc": acc}

    int/floatのstep, logはoptional。

    """

    def params_fn(trial):
        table = {
            "categorical": trial.suggest_categorical,
            "discrete_uniform": trial.suggest_discrete_uniform,
            "float": trial.suggest_float,
            "int": trial.suggest_int,
            "loguniform": trial.suggest_loguniform,
            "uniform": trial.suggest_uniform,
        }
        return {
            key: table[suggest](name=key, **kwargs)
            for key, (suggest, kwargs) in params.items()
        }

    def score_fn(params):
        evals = func(**params)
        return evals[score_name]

    tk.hpo.optimize(params_fn, score_fn, direction=direction, n_trials=n_trials)
