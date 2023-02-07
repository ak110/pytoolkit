"""テストコード"""
import pathlib

import pytoolkit
import pytoolkit.tables

test_data_path = pathlib.Path(__file__).parent.parent / "test_data" / "iris.csv"


def test_bigbang(tmpdir):
    """まとめて全部テスト"""
    model_dir = tmpdir

    # 学習
    train_data, train_labels = pytoolkit.tables.load_labeled_data(
        test_data_path, "variety"
    )
    model = pytoolkit.tables.train(train_data, train_labels, groups=None)
    model.save(model_dir)

    # 検証
    test_data, test_labels = pytoolkit.tables.load_labeled_data(
        test_data_path, "variety"
    )
    model = pytoolkit.tables.load(model_dir)
    score = model.evaluate(test_data, test_labels)
    assert 0.99 <= score["auc"] <= 1.0

    # 推論
    input_data = pytoolkit.tables.load_unlabeled_data(test_data_path)
    input_data.drop_in_place("variety")
    model = pytoolkit.tables.load(model_dir)
    results = model.infer(input_data)
    assert len(input_data) == len(results)
