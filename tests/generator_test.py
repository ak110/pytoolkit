
import numpy as np

import pytoolkit as tk


def test_generator():
    generator = tk.generator.Generator()

    # シャッフル無しならバッチサイズが変わってぴったり一周になることを確認
    g, steps = generator.flow(np.array([1, 2, 3]), batch_size=2, shuffle=False)
    assert steps == 2
    assert (next(g) == np.array([1, 2])).all()
    assert (next(g) == np.array([3])).all()
    assert (next(g) == np.array([1, 2])).all()
    g.close()

    # シャッフルありなら毎回同じバッチサイズであることを確認
    g, steps = generator.flow(np.array([1, 2, 3]), batch_size=2, shuffle=True)
    assert steps == 2
    assert next(g).shape == (2,)
    assert next(g).shape == (2,)
    assert next(g).shape == (2,)
    g.close()
