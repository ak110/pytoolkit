import os
import pathlib
import pytest

import pytoolkit as tk


def test_moving_average():
    ma = tk.moving_average([1, 2, 3, 4, 5], 3)
    assert ma[0] == pytest.approx((1 + 2 + 3) / 3)
    assert ma[1] == pytest.approx((2 + 3 + 4) / 3)
    assert ma[2] == pytest.approx((3 + 4 + 5) / 3)
    assert len(ma) == 3


def test_get_gpu_count():
    assert tk.get_gpu_count() >= 0


def test_create_gpu_pool(tmpdir):
    with tk.create_gpu_pool(1, processes_per_gpu=3) as p:
        p.map(_test_create_gpu_pool_func, [(str(tmpdir), i) for i in range(3)])

    assert pathlib.Path(str(tmpdir.join('0.log'))).is_file()
    assert pathlib.Path(str(tmpdir.join('1.log'))).is_file()
    assert pathlib.Path(str(tmpdir.join('2.log'))).is_file()
    assert len(tmpdir.listdir()) == 3


def _test_create_gpu_pool_func(arg):
    tmpdir, a = arg
    assert a in range(3)
    pid = os.environ['GPU_POOL_PID']
    print(f'a={a} pid={pid}')
    with open(os.path.join(tmpdir, f'{a}.log'), 'w') as f:
        f.write(f'{a}')
