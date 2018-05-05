import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_load_voc_file():
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    xml_path = data_dir / 'VOC2007_000001.xml'
    ann = tk.data.voc.load_annotation(data_dir, xml_path)
    assert ann.path == data_dir / 'VOC2007' / 'JPEGImages' / '000001.jpg'
    assert ann.width == 353
    assert ann.height == 500
    assert len(ann.classes) == 2
    assert ann.classes[0] == tk.data.voc.CLASS_NAMES_TO_ID['dog']
    assert ann.classes[1] == tk.data.voc.CLASS_NAMES_TO_ID['person']
    assert (ann.difficults == np.array([False, False])).all()
    assert ann.bboxes[0] == pytest.approx(np.array([48, 240, 195, 371]) / [353, 500, 353, 500])
    assert ann.bboxes[1] == pytest.approx(np.array([8, 12, 352, 498]) / [353, 500, 353, 500])
