import os
import pathlib
import pytest

import pytoolkit as tk


def test_create_tee_logger(tmpdir):
    log_path = str(tmpdir.join('test.log'))
    logger = tk.create_tee_logger(log_path, name='test')
    logger.debug('あいうえお')
    with open(log_path, encoding='utf-8') as f:
        assert f.read() == 'あいうえお\n'

    tk.close_logger(logger)
    os.remove(log_path)


def test_memorized(tmpdir):
    cache_path = str(tmpdir.join('cache.pkl'))
    assert tk.memorized(cache_path, lambda: 1) == 1  # fnの結果
    assert tk.memorized(cache_path, lambda: 2) == 1  # cacheされた結果


def test_moving_average():
    ma = tk.moving_average([1, 2, 3, 4, 5], 3)
    assert ma[0] == pytest.approx((1 + 2 + 3) / 3)
    assert ma[1] == pytest.approx((2 + 3 + 4) / 3)
    assert ma[2] == pytest.approx((3 + 4 + 5) / 3)
    assert len(ma) == 3


def test_get_gpu_count():
    assert tk.get_gpu_count() >= 1


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
    tid = os.environ['GPU_POOL_PID']
    with open(os.path.join(tmpdir, '{}.log'.format(tid)), 'w') as f:
        f.write('{}'.format(tid))
