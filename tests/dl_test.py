
import numpy as np

import pytoolkit as tk


def test_generator():
    gen = tk.dl.Generator()

    assert gen.steps_per_epoch(3, 2) == 2

    g = gen.flow(np.array([1, 2, 3]), batch_size=2, shuffle=False)
    assert (g.__next__() == np.array([1, 2])).all()
    assert (g.__next__() == np.array([3])).all()
    assert (g.__next__() == np.array([1, 2])).all()
    g.close()

    g = gen.flow(np.array([1, 2, 3]), batch_size=2, shuffle=True)
    assert g.__next__().shape == (2,)
    assert g.__next__().shape == (2,)
    assert g.__next__().shape == (2,)
    g.close()
