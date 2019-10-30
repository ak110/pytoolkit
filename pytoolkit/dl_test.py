import pytoolkit as tk


def test_get_gpu_count():
    assert tk.dl.get_gpu_count() >= 0
