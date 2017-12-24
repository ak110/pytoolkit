import io
import os
import pathlib

import pytest

import pytoolkit as tk


def test_logger(tmpdir):
    log_path = str(tmpdir.join('test.log'))
    stderr = io.StringIO()

    logger = tk.log.get('test_logger')
    logger.addHandler(tk.log.stream_handler(stderr, fmt=None))
    logger.addHandler(tk.log.file_handler(log_path, fmt=None))

    logger.info('あいうえお')
    assert stderr.getvalue() == 'あいうえお\n'
    with open(log_path, encoding='utf-8') as f:
        assert f.read() == 'あいうえお\n'

    tk.log.close(logger)
    os.remove(log_path)


def test_trace():
    import logging
    stderr = io.StringIO()
    logger = tk.log.get('test_trace')
    logger.addHandler(tk.log.stream_handler(stderr, level=logging.DEBUG, fmt=None))

    @tk.log.trace('test_trace')
    def _traced_func():
        logger.debug('あいうえお')

    _traced_func()
    _traced_func()
    tk.log.close(logger)

    lines = stderr.getvalue().split('\n')
    assert lines[0] == '_traced_func 開始'
    assert lines[1] == 'あいうえお'
    assert lines[2].startswith('_traced_func 終了 (time=')
    assert lines[3] == '_traced_func 開始'
    assert lines[4] == 'あいうえお'
    assert lines[5].startswith('_traced_func 終了 (time=')
