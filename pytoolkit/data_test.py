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
        def get_sample(self, data):
            assert len(data) == 2
            data1, data2 = data
            assert isinstance(data1, tuple) and len(data1) == 2  # X, y
            assert isinstance(data2, tuple) and len(data2) == 2  # X, y
            return data1

    dataset = tk.data.Dataset(data=np.arange(3), labels=np.arange(4, 7))
    data_loader = MyDataLoader(batch_size=2, data_per_sample=2)
    iterator = data_loader.iter(dataset, shuffle=True)
    g = iter(iterator.ds)
    for _ in range(3):
        X_batch, y_batch = next(g)
        assert X_batch.numpy().shape == (2,) and y_batch.numpy().shape == (2,)


@pytest.mark.parametrize("data_per_sample", [1, 2, 3])
def test_data_loader_dict_and_none(data_per_sample):
    """X=dict(), y=Noneのケース"""

    class MyDataLoader(tk.data.DataLoader):
        def get_sample(self, data):
            assert len(data) == data_per_sample
            assert isinstance(data[0], tuple) and len(data[0]) == 2  # X, y
            return data[0]

    data = np.array(
        [{"a": np.zeros(()), "b": np.zeros(2)}, {"a": np.ones(()), "b": np.ones(2)}]
    )
    labels = None
    dataset = tk.data.Dataset(data=data, labels=labels)
    data_loader = MyDataLoader(batch_size=2, data_per_sample=data_per_sample)
    iterator = data_loader.iter(dataset, shuffle=False)
    g = iter(iterator.ds)

    X_batch, y_batch = next(g)
    if data_per_sample in (1, 3):
        assert X_batch["a"].numpy() == pytest.approx(np.array([0, 1]))
        assert X_batch["b"].numpy() == pytest.approx(np.array([[0, 0], [1, 1]]))
    else:
        assert X_batch["a"].numpy() == pytest.approx(np.array([0, 0]))
        assert X_batch["b"].numpy() == pytest.approx(np.array([[0, 0], [0, 0]]))
    assert y_batch.numpy() == pytest.approx(np.array([0, 0]))
