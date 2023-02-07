"""テストコード。"""

import pytoolkit


def test_log(caplog):
    caplog.set_level("INFO")

    logger = pytoolkit.logs.get("test")
    with pytoolkit.logs.timer("timer", logger):
        logger.warning("warn")

    assert caplog.records[0].message == "<timer> start."
    assert caplog.records[1].message == "warn"
    assert caplog.records[2].message == "<timer> done in 0 s."
