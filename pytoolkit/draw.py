"""主にmatplotlib関連。

出来るだけobject-oriented interfaceを使用する。

例::

    ax = df.plot()  # type: matplotlib.axes.Axes

"""
import io
import pathlib
import threading

import numpy as np

from . import utils

_lock = threading.Lock()


def get_lock() -> threading.Lock:
    """Flaskなどでマルチスレッドで使う場合用のロックを返す。

    matplotlib周りは構造上たぶんスレッドセーフじゃなさそうなので(?)、
    使う側で適当にこれでロックして使うことにしてみる。
    """
    return _lock


def create_figure(figsize=None, dpi=None, facecolor=None, edgecolor=None,
                  linewidth=0.0, frameon=None, subplotpars=None, tight_layout=None, **kwargs):
    """Figureを作って返す。"""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig = matplotlib.figure.Figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor,
                                   linewidth=linewidth, frameon=frameon,
                                   subplotpars=subplotpars, tight_layout=tight_layout, **kwargs)
    FigureCanvas(fig)
    return fig


@utils.memoize
def get_colors(count, cmap='hsv', scale=255):
    """色を列挙する。

    Args:
        count: 色の個数
        cmap: matplotlibのカラーマップの名前。
        scale: 1なら0～1、255なら0～255で返す。

    Returns:
        (count, 4)のndarray。4はRGBA。

    """
    import matplotlib.cm
    return matplotlib.cm.get_cmap(name=cmap)(np.linspace(0, 1, count + 1)[:count]) * scale


def set_axis_tick_to_int(axis):
    """軸のラベルを整数のみにする。

    使用例::

        ax.set_xlabel('Epochs')
        tk.draw.set_axis_tick_to_int(ax.get_xaxis())

    """
    import matplotlib
    axis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def save_to_bytes(ax, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format=None,  # pylint: disable=W0622
                  transparent=False, bbox_inches=None, pad_inches=0.1,
                  frameon=None, **kwargs) -> bytes:
    """bytesに保存。"""
    with io.BytesIO() as f:
        save(ax, f, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor,
             orientation=orientation, papertype=papertype, format=format,
             transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches,
             frameon=frameon, **kwargs)
        return f.getvalue()


def save(ax, file, dpi=None, facecolor='w', edgecolor='w',
         orientation='portrait', papertype=None, format=None,  # pylint: disable=W0622
         transparent=False, bbox_inches=None, pad_inches=0.1,
         frameon=None, **kwargs):
    """保存。"""
    # ディレクトリ作成
    if isinstance(file, (str, pathlib.Path)):
        file = pathlib.Path(file)
        file.resolve().parent.mkdir(parents=True, exist_ok=True)
        if format is None:
            format = file.suffix[1:]
    else:
        if format is None:
            format = 'png'
    # 保存
    ax.get_figure().savefig(
        file, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor,
        orientation=orientation, papertype=papertype, format=format,
        transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches,
        frameon=frameon, **kwargs)


def close(ax):
    """後始末。"""
    # …これで正しいかは不明…
    import matplotlib.pyplot as plt
    plt.close(ax.get_figure())
