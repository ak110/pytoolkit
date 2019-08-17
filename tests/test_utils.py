import pytoolkit as tk


def test_memoize():
    count = [0]

    @tk.utils.memoize
    def f(a, b):
        count[0] += 1
        return a + b["c"]

    assert f(1, b={"c": 2}) == 3
    assert f(1, b={"c": 3}) == 4
    assert f(1, b={"c": 2}) == 3
    assert count[0] == 2
