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


def test_format_exc():
    try:
        a = "test"
        raise RuntimeError(a)
    except RuntimeError:
        s = tk.utils.format_exc(safe=False)
        assert s != ""


def test_tqdm():
    a = ["a", "b", "c"]
    assert tuple(tk.utils.tqdm(a)) == tuple(a)


def test_trange():
    assert tuple(tk.utils.trange(3)) == tuple(range(3))


def test_tenumerate():
    a = ["a", "b", "c"]
    assert tuple(tk.utils.tenumerate(a)) == tuple(enumerate(a))


def test_tzip():
    a = ["a", "b", "c"]
    b = ["d", "e", "f"]
    assert tuple(tk.utils.tzip(a, b)) == tuple(zip(a, b))


def test_tmap():
    def f(x):
        return x * 2

    assert tuple(tk.utils.tmap(f, range(3))) == tuple(map(f, range(3)))
