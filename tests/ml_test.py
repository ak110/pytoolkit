import pathlib
import pytoolkit as tk


def test_plot_cm(tmpdir):
    filepath = str(tmpdir.join('confusion_matrix.png'))
    cm = [
        [5, 0, 0],
        [0, 3, 2],
        [0, 2, 3],
    ]
    tk.ml.plot_cm(cm, filepath)
    assert pathlib.Path(filepath).is_file()  # とりあえずファイルの存在チェックだけ。。
