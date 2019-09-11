"""回帰の評価。"""
import numpy as np
import sklearn.metrics

import pytoolkit as tk


def print_regression_metrics(y_true, y_pred, print_fn=None):
    """回帰の指標色々を表示する。"""
    try:
        print_fn = print_fn or tk.log.get(__name__).info
        y_mean = np.tile(np.mean(y_pred), len(y_true))
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
        rmseb = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_mean))
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        maeb = sklearn.metrics.mean_absolute_error(y_true, y_mean)
        print_fn(f"R^2:      {r2:.3f}")
        print_fn(f"RMSE:     {rmse:.3f} (base: {rmseb:.3f})")
        print_fn(f"MAE:      {mae:.3f} (base: {maeb:.3f})")
        # RMSE/MAEが1.253より小さいか大きいかで分布の予想がちょっと出来る
        # https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/#how-to-check-rmse-mae-summary
        print_fn(f"RMSE/MAE: {rmse / mae:.3f}")
    except BaseException:
        tk.log.get(__name__).warning("Error: print_regression_metrics", exc_info=True)
