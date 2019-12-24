import numpy as np
import pytest

import pytoolkit as tk


def test_data_loader():
    dataset = tk.data.Dataset(data=np.arange(3), labels=np.arange(4, 7))
    data_loader = tk.data.DataLoader(batch_size=2, data_per_sample=1)
    iterator = data_loader.iter(dataset, shuffle=False)
    g = iter(iterator.ds)

    X_batch, y_batch = next(g)
    assert (X_batch.numpy(), y_batch.numpy()) == (
        pytest.approx(np.array([0, 1])),
        pytest.approx(np.array([4, 5])),
    )

    X_batch, y_batch = next(g)
    assert (X_batch.numpy(), y_batch.numpy()) == (
        pytest.approx(np.array([2])),
        pytest.approx(np.array([6])),
    )

    with pytest.raises(StopIteration):
        next(g)


def test_data_loader_2():
    """data_per_sample=2のケース"""

    class MyDataLoader(tk.data.DataLoader):
        def get_sample(self, data: list) -> tuple:
            assert len(data) == 2
            return data[0]

    dataset = tk.data.Dataset(data=np.arange(3), labels=np.arange(4, 7))
    data_loader = MyDataLoader(batch_size=2, data_per_sample=2)
    iterator = data_loader.iter(dataset, shuffle=True)
    g = iter(iterator.ds)
    for _ in range(3):
        X_batch, y_batch = next(g)
        assert X_batch.shape == (2,) and y_batch.shape == (2,)


def test_data_loader_dict_and_none():
    """X=dict(), y=Noneのケース"""
    pytest.skip("作業中。。")

    data = np.array(
        [{"a": np.zeros(()), "b": np.zeros(2)}, {"a": np.ones(()), "b": np.ones(2)}]
    )
    labels = None
    dataset = tk.data.Dataset(data=data, labels=labels)
    data_loader = tk.data.DataLoader(batch_size=2, data_per_sample=1)
    iterator = data_loader.iter(dataset, shuffle=False)
    g = iter(iterator.ds)

    X_batch, y_batch = next(g)
    assert X_batch.numpy() == pytest.approx(data)
    assert y_batch.numpy() == pytest.approx([0, 0])

    with pytest.raises(StopIteration):
        next(g)
