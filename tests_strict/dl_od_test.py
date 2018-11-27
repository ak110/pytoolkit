import pathlib

import numpy as np

import pytoolkit as tk


def test_od(tmpdir):
    result_dir = pathlib.Path(str(tmpdir))

    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'tests' / 'data' / 'od'
    class_name_to_id = {'～': 0, '〇': 1}
    X, y = tk.data.voc.load_annotations(data_dir, data_dir / 'Annotations', class_name_to_id=class_name_to_id)
    X = np.array([data_dir / 'JPEGImages' / (p.stem + '.png') for p in X])  # TODO: VoTT対応

    with tk.dl.session():
        od = tk.dl.od.ObjectDetector((128, 128), [16, 8], 2)
        od.fit(X, y, X, y,
               batch_size=3, epochs=1,
               initial_weights=None,
               pb_size_pattern_count=8,
               flip_h=True, flip_v=False, rotate90=False,
               plot_path=result_dir / 'model.svg',
               tsv_log_path=result_dir / 'history.tsv')
        od.save(result_dir / 'model.json')
        od.save_weights(result_dir / 'model.h5')

    with tk.dl.session():
        od = tk.dl.od.ObjectDetector.load(result_dir / 'model.json')
        od.load_weights(result_dir / 'model.h5', batch_size=3, strict_nms=True, use_multi_gpu=True)
        pred = od.predict(X, conf_threshold=0.25)
        assert len(pred) == len(y)
