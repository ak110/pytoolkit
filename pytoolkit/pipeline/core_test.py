from __future__ import annotations

import numpy as np
import pytest

import pytoolkit as tk


@pytest.mark.parametrize("output_count", [1, 2])
def test_predict(output_count, tmpdir):
    # pylint: disable=abstract-method
    dataset = tk.data.Dataset(data=np.random.randint(0, 256, size=(3, 2, 1)))
    folds = [([0, 1], [2]), ([1, 2], [0]), ([2, 0], [1])]

    class TestModel(tk.pipeline.Model):
        def _predict(
            self, dataset: tk.data.Dataset, fold: int
        ) -> np.ndarray | list[np.ndarray]:
            if output_count == 1:
                return dataset.data
            else:
                return [dataset.data, np.array([fold] * len(dataset))]

    model = TestModel(nfold=len(folds), models_dir=str(tmpdir))

    # predict_all
    result = model.predict_all(dataset)
    assert len(result) == len(folds)
    if output_count == 1:
        assert (result[0] == dataset.data).all()
        assert (result[1] == dataset.data).all()
        assert (result[2] == dataset.data).all()
    else:
        assert (result[0][0] == dataset.data).all()
        assert (result[1][0] == dataset.data).all()
        assert (result[2][0] == dataset.data).all()
        assert (result[0][1] == np.array([0, 0, 0])).all()
        assert (result[1][1] == np.array([1, 1, 1])).all()
        assert (result[2][1] == np.array([2, 2, 2])).all()

    # predict_oof
    result = model.predict_oof(dataset, folds)
    if output_count == 1:
        assert (result == dataset.data).all()
    else:
        assert len(result) == 2
        assert (result[0] == dataset.data).all()
        assert (result[1] == np.array([1, 2, 0])).all()
