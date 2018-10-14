import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_load_voc_file():
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'data' / 'od'
    xml_path = data_dir / 'Annotations' / '無題.xml'
    class_name_to_id = {'～': 0, '〇': 1}
    ann = tk.data.voc.load_annotation(data_dir, xml_path, class_name_to_id)
    assert ann.path == (data_dir / 'Annotation' / 'JPEGImages' / '無題')  # TODO: VoTT対応
    assert ann.width == 768
    assert ann.height == 614
    assert len(ann.classes) == 1
    assert ann.classes[0] == 0
    assert (ann.difficults == np.array([False])).all()
    assert ann.bboxes[0] == pytest.approx(np.array([203, 255, 601, 355]) / [768, 614, 768, 614])
