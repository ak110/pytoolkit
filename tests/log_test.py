import io
import os

import pytoolkit as tk


def test_init(tmpdir):
    log_path = str(tmpdir.join('test.log'))
    tk.log.init(log_path, stream_fmt=None, file_fmt=None)

    logger = tk.log.get(None)
    logger.debug('あいうえお')
    with open(log_path, encoding='utf-8') as f:
        assert f.read() == 'あいうえお\n'

    tk.log.close(logger)
    os.remove(log_path)


def test_logger(tmpdir):
    log_path = str(tmpdir.join('test.log'))
    stderr = io.StringIO()

    logger = tk.log.get('test_logger')
    logger.addHandler(tk.log.stream_handler(stderr, level='INFO', fmt=None))
    logger.addHandler(tk.log.file_handler(log_path, level='DEBUG', fmt=None))

    logger.info('あいうえお')
    assert stderr.getvalue() == 'あいうえお\n'
    with open(log_path, encoding='utf-8') as f:
        assert f.read() == 'あいうえお\n'

    tk.log.close(logger)
    os.remove(log_path)


def test_trace():
    import logging
    stderr = io.StringIO()
    logger = tk.log.get(tk.log.__name__)
    tk.log.close(logger)
    logger.addHandler(tk.log.stream_handler(stderr, level=logging.DEBUG, fmt=None))

    @tk.log.trace()
    def _traced_func():
        logger.debug('あいうえお')

    _traced_func()
    _traced_func()
    tk.log.close(logger)

    lines = stderr.getvalue().split('\n')
    assert lines[0] == 'test_trace.<locals>._traced_func 開始'
    assert lines[1] == 'あいうえお'
    assert lines[2].startswith('test_trace.<locals>._traced_func 終了 (')
    assert lines[3] == 'test_trace.<locals>._traced_func 開始'
    assert lines[4] == 'あいうえお'
    assert lines[5].startswith('test_trace.<locals>._traced_func 終了 (')