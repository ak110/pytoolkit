
import pytoolkit as tk


def test_get_gpu_count():
    assert tk.utils.get_gpu_count() >= 0
