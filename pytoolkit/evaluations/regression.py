"""回帰の評価。"""
import typing

import numpy as np
import sklearn.metrics

import pytoolkit as tk


def print_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    print_fn: typing.Callable[[str], None] = None,
) -> typing.Dict[str, typing.Any]:
    """回帰の指標色々を表示する。"""
    try:
        evals = evaluate_regression(y_true, y_pred)
        print_fn = print_fn or tk.log.get(__name__).info
        print_fn(f"R^2:      {evals['r2']:.3f}")
        print_fn(f"RMSE:     {evals['rmse']:.3f} (base: {evals['rmse_base']:.3f})")
        print_fn(f"MAE:      {evals['mae']:.3f} (base: {evals['mae_base']:.3f})")
        print_fn(f"RMSE/MAE: {evals['rmse/mae']:.3f}")
        return evals
    except BaseException:
        tk.log.get(__name__).warning("Error: print_regression_metrics", exc_info=True)
        return {}


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> typing.Dict[str, typing.Any]:
    """回帰の指標色々を算出してdictで返す。"""
    y_mean = np.tile(np.mean(y_pred), len(y_true))
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    rmseb = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_mean))
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    maeb = sklearn.metrics.mean_absolute_error(y_true, y_mean)
    return {
        "r2": r2,
        "rmse": rmse,
        "rmse_base": rmseb,
        "mae": mae,
        "mae_base": maeb,
        # RMSE/MAEが1.253より小さいか大きいかで分布の予想がちょっと出来る
        # https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/#how-to-check-rmse-mae-summary
        "rmse/mae": rmse / mae,
    }
