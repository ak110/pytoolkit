import pathlib
import numpy as np

import pytoolkit as tk


def test_ic(tmpdir):
    models_dir = pathlib.Path(str(tmpdir))

    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'tests' / 'data' / 'ic'
    class_names, X, y = tk.ml.listup_classification(data_dir, check_image=True)

    model = tk.dl.ic.ImageClassifier.create(class_names, 'vgg16bn')
    model.fit(X, y, validation_data=(X, y),
              epochs=1,
              tsv_log_path=models_dir / 'history.tsv',
              mixup=True, cosine_annealing=True)
    model.save(models_dir / 'model.h5', include_optimizer=False)

    model = tk.dl.ic.ImageClassifier.load(models_dir / 'model.h5', 1)
    assert class_names == model.class_names
    pred = model.predict(X)
    assert len(pred) == len(y)
