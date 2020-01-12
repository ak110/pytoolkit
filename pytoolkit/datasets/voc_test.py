import numpy as np
import pytest

import pytoolkit as tk


def test_load_voc_od_split(data_dir):
    ds = tk.datasets.load_voc_od_split(data_dir / "od", split="train")
    assert len(ds) == 3
    assert tuple(ds.metadata["class_names"]) == ("～", "〇")

    ann = ds.labels[0]
    assert ann.path == (data_dir / "od" / "JPEGImages" / "無題.jpg")
    assert ann.width == 768
    assert ann.height == 614
    assert len(ann.classes) == 1
    assert ann.classes[0] == 0
    assert (ann.difficults == np.array([False])).all()
    assert ann.bboxes[0] == pytest.approx(
        np.array([203 - 1, 255 - 1, 601 - 1, 355 - 1]) / [768, 614, 768, 614]
    )
