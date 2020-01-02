import io
import os

import pytoolkit as tk


def test_init(tmpdir):
    log_path = str(tmpdir.join("test.log"))
    tk.log.init(log_path, stream_fmt=None, file_fmt=None)

    logger = tk.log.get(None)
    logger.debug("あいうえお")
    with open(log_path, encoding="utf-8") as f:
        assert f.read() == "あいうえお\n"

    tk.log.close(logger)
    os.remove(log_path)


def test_logger(tmpdir):
    log_path = str(tmpdir.join("test.log"))
    stderr = io.StringIO()

    logger = tk.log.get("test_logger")
    logger.addHandler(tk.log.stream_handler(stderr, level="INFO", fmt=None))
    logger.addHandler(tk.log.file_handler(log_path, level="DEBUG", fmt=None))

    logger.info("あいうえお")
    assert stderr.getvalue() == "あいうえお\n"
    with open(log_path, encoding="utf-8") as f:
        assert f.read() == "あいうえお\n"

    tk.log.close(logger)
    os.remove(log_path)


def test_trace():
    stderr = io.StringIO()
    logger = tk.log.get(tk.log.__name__)
    tk.log.close(logger)
    logger.addHandler(tk.log.stream_handler(stderr, level="DEBUG", fmt=None))

    with tk.log.trace_scope("trace_scope"):
        logger.debug("あいうえお")

    tk.log.close(logger)

    lines = stderr.getvalue().split("\n")
    assert lines[0] == "trace_scope start"
    assert lines[1] == "あいうえお"
    assert lines[2].startswith("trace_scope done in ")
