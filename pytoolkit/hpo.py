"""Optuna関連など。"""
from __future__ import annotations

import inspect
import logging
import typing

if typing.TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


def optimize(
    params_fn: typing.Callable[[optuna.trial.BaseTrial], typing.Any],
    score_fn: typing.Callable[..., float],
    storage=None,
    sampler=None,
    pruner=None,
    study_name=None,
    direction="minimize",
    load_if_exists=False,
    n_trials=None,
    timeout=None,
    n_jobs=1,
    catch=(Exception,),
) -> optuna.study.Study:
    """Optunaの簡易ラッパー。

    Args:
        params_fn: trialを受け取り、dictなどを返す関数。
        score_fn: params_fnの結果と、trial(省略可)を受け取り、スコアを返す関数。
                  (trialを受け取りたい場合は引数名は必ず`trial`にする。)
        storage: optuna.create_studyの引数
        sampler: optuna.create_studyの引数
        pruner: optuna.create_studyの引数
        study_name: optuna.create_studyの引数
        direction: optuna.create_studyの引数
        load_if_exists: optuna.create_studyの引数
        n_trials: study.optimizeの引数
        timeout: study.optimizeの引数
        n_jobs: study.optimizeの引数
        catch: study.optimizeの引数

    Returns:
        study object

    suggest_*メモ:
        - trial.suggest_categorical(name, choices)
        - trial.suggest_discrete_uniform(name, low, high, q)
        - trial.suggest_float(name, low, high, step=None, log=False)  # [low, high]
        - trial.suggest_int(name, low, high, step=1, log=False)  # [low, high]
        - trial.suggest_loguniform(name, low, high)  # [low,high)
        - trial.suggest_uniform(name, low, high)  # [low,high)

    References:
        - <https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.create_study>
        - <https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.Study.optimize>

    """
    import optuna  # pylint: disable=redefined-outer-name

    def objective(trial):
        params = params_fn(trial)
        if "trial" in inspect.signature(score_fn).parameters:
            value = score_fn(params, trial=trial)
        else:
            value = score_fn(params)
        logger.debug(f"value = {value}, params = {params}")
        return value

    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )
    try:
        study.optimize(
            func=objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=catch,
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
    finally:
        if len(study.trials) > 0:
            # params_fnの戻り値を最終結果としてログ出力
            logger.info(f"best value = {study.best_value}")
            logger.info(f"best params = {get_best_params(study, params_fn)}")

    return study


def raise_pruned() -> typing.NoReturn:
    """`raise optuna.exceptions.TrialPruned()` する。(params_fn/score_fnから呼び出す用)"""
    import optuna  # pylint: disable=redefined-outer-name

    raise optuna.exceptions.TrialPruned()


def get_best_params(
    study: optuna.study.Study,
    params_fn: typing.Callable[[optuna.trial.BaseTrial], typing.Any] = None,
) -> dict[str, float]:
    """見つけた中で最善のパラメータを返す。"""
    import optuna  # pylint: disable=redefined-outer-name

    if params_fn is None:
        return study.best_params
    else:
        return params_fn(optuna.trial.FixedTrial(study.best_params))
