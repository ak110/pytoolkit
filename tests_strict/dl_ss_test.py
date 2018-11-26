import pathlib
import numpy as np

import pytoolkit as tk


def test_ss_multi(tmpdir):
    models_dir = pathlib.Path(str(tmpdir))

    X = np.zeros((2, 32, 32, 3))
    y = np.random.randint(0, 256, size=X.shape)
    class_colors = [
        (0, 0, 0),
        (0, 0, 128),
        (0, 128, 0),
        (128, 0, 0),
        (128, 128, 128),
    ]
    void_color = (255, 255, 255)

    with tk.dl.session():
        model = tk.dl.ss.SemanticSegmentor.create(
            class_colors, void_color, input_size=64, weights=None)
        model.fit(X, y, validation_data=(X, y),
                  epochs=1,
                  tsv_log_path=models_dir / 'history.tsv',
                  mixup=True, cosine_annealing=True)
        model.save(models_dir / 'model.h5', include_optimizer=False)

    with tk.dl.session():
        model = tk.dl.ss.SemanticSegmentor.load(models_dir / 'model.h5')
        pred = model.predict(X)
        assert len(pred) == len(y)
        ious, miou = model.compute_mean_iou(y, pred)
        assert len(ious) == len(class_colors)
        assert 0 <= miou <= 1
