
import numpy as np

import pytoolkit as tk


def test_generator():
    gen = tk.generator.Generator()

    # シャッフル無しならバッチサイズが変わってぴったり一周になることを確認
    seq = iter(gen.flow(np.array([1, 2, 3]), batch_size=2, shuffle=False))
    assert (next(seq) == np.array([1, 2])).all()
    assert (next(seq) == np.array([3])).all()
    assert (next(seq) == np.array([1, 2])).all()
    seq.close()

    # シャッフルありなら毎回同じバッチサイズであることを確認
    seq = iter(gen.flow(np.array([1, 2, 3]), batch_size=2, shuffle=True))
    assert next(seq).shape == (2,)
    assert next(seq).shape == (2,)
    assert next(seq).shape == (2,)
    seq.close()
