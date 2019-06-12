import pytest

import pytoolkit as tk

K = tk.K


def test_logit(session):
    x = K.constant([0.0, 0.5, 1.0])
    y = [-16.118095, 0, +16.118095]
    assert session.run(tk.backend.logit(x)) == pytest.approx(y, abs=1e-6)
