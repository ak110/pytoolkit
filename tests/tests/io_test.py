import pathlib

import pytest

import pytoolkit as tk


def test_read_write(tmpdir):
    path = pathlib.Path(tmpdir.join('read_write_test.txt'))

    path.write_text('あああ\nいいい', encoding='utf-8')
    assert path.read_text(encoding='utf-8') == 'あああ\nいいい'
    assert tk.io.read_all_lines(path) == ['あああ', 'いいい']

    tk.io.write_all_lines(path, ['ううう', 'えええ'])
    assert path.read_text(encoding='utf-8') == 'ううう\nえええ\n'
    assert tk.io.read_all_lines(path) == ['ううう', 'えええ']


def test_do_retry():
    d = {'count1': 0, 'count2': 0}

    def _func1():
        d['count1'] += 1
        return d['count1']

    def _func2():
        d['count2'] += 1
        if d['count2'] <= 3:
            raise RuntimeError()
        return d['count2']

    assert tk.io.do_retry(_func1, sleep_seconds=0) == 1
    assert tk.io.do_retry(_func2, sleep_seconds=0) == 4
    with pytest.raises(ZeroDivisionError):
        tk.io.do_retry(lambda: 1 // 0, sleep_seconds=0)
