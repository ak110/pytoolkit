"""PASCAL VOCデータセット関連。

以下の3ファイルを解凍して出来たVOCdevkitディレクトリのパスを受け取って色々処理をする。

- [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- [VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- [VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

"""
from __future__ import annotations

import pathlib
import typing

import numpy as np

import pytoolkit as tk

# VOC2007のクラス名のリスト (20クラス)
CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def load_voc_od(
    voc_dir: tk.typing.PathLike, verbose: bool = True,
) -> typing.Tuple[tk.data.Dataset, tk.data.Dataset]:
    """PASCAL VOCの物体検出のデータを読み込む。(07+12 trainval / 07 test)

    References:
        - <https://chainercv.readthedocs.io/en/stable/reference/datasets.html#vocbboxdataset>

    """
    from chainercv.datasets.voc.voc_bbox_dataset import VOCBboxDataset

    voc_dir = pathlib.Path(voc_dir)
    ds_train07 = _load_from_chainercv(
        VOCBboxDataset(
            data_dir=str(voc_dir / "VOC2007"),
            split="trainval",
            year="2007",
            use_difficult=False,
        ),
        desc="load VOC 07 trainval",
        verbose=verbose,
    )
    ds_train12 = _load_from_chainercv(
        VOCBboxDataset(
            data_dir=str(voc_dir / "VOC2012"),
            split="trainval",
            year="2012",
            use_difficult=False,
        ),
        desc="load VOC 12 trainval",
        verbose=verbose,
    )
    ds_val = _load_from_chainercv(
        VOCBboxDataset(
            data_dir=str(voc_dir / "VOC2007"),
            split="test",
            year="2007",
            use_difficult=True,
            return_difficult=True,
        ),
        desc="load VOC 07 test",
        verbose=verbose,
    )
    return tk.data.Dataset.concat(ds_train07, ds_train12), ds_val


def _load_from_chainercv(ds, desc, verbose) -> tk.data.Dataset:
    labels = np.array(
        [
            _get_label(ds, i)
            for i in tk.utils.trange(len(ds), desc=desc, disable=not verbose)
        ]
    )
    data = np.array([y.path for y in labels])
    return tk.data.Dataset(
        data=data, labels=labels, metadata={"class_names": CLASS_NAMES}
    )


def _get_label(ds, i: int) -> tk.od.ObjectsAnnotation:
    # pylint: disable=protected-access

    # https://github.com/chainer/chainercv/blob/fddc813/chainercv/datasets/voc/voc_bbox_dataset.py#L84
    path = pathlib.Path(ds.data_dir) / "JPEGImages" / f"{ds.ids[i]}.jpg"

    height, width = tk.ndimage.get_image_size(path)

    bboxes, classes, difficults = ds._get_annotations(i)

    # (ymin,xmin,ymax,xmax) -> (xmin,ymin,xmax,ymax)
    bboxes = bboxes[:, [1, 0, 3, 2]].astype(np.float32)
    bboxes[:, [0, 2]] /= width
    bboxes[:, [1, 3]] /= height

    return tk.od.ObjectsAnnotation(
        path=path,
        width=width,
        height=height,
        classes=classes,
        bboxes=bboxes,
        difficults=difficults,
    )
