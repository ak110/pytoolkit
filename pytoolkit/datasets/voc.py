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
    voc_dir: tk.typing.PathLike,
) -> typing.Tuple[tk.data.Dataset, tk.data.Dataset]:
    """PASCAL VOCの物体検出のデータを読み込む。(07+12 trainval / 07 test)

    Examples:
        train_set, val_set = tk.datasets.load_voc_od("/path/to/VOCdevkit")

    """
    voc_dir = pathlib.Path(voc_dir)
    ds_train07 = load_voc_od_split(
        data_dir=voc_dir / "VOC2007",
        split="trainval",
        use_difficult=False,
    )
    ds_train12 = load_voc_od_split(
        data_dir=voc_dir / "VOC2012",
        split="trainval",
        use_difficult=False,
    )
    ds_val = load_voc_od_split(
        data_dir=voc_dir / "VOC2007",
        split="test",
        use_difficult=True,
    )
    return tk.data.Dataset.concat(ds_train07, ds_train12), ds_val


def load_voc_od_split(
    data_dir: tk.typing.PathLike, split: str, use_difficult: bool = True
):
    """VOC形式の物体検出のデータの読み込み。"""
    import xml.etree.ElementTree as ET

    assert split in ("train", "trainval", "val", "test")
    data_dir = pathlib.Path(data_dir)

    class_names = CLASS_NAMES
    # pascal_label_map.pbtxtがあればそこからクラス名＋IDを読み込み
    label_map_path = data_dir / "pascal_label_map.pbtxt"
    if label_map_path.exists():
        # 本当は google.protobuf を使うべきだが、雑に読み込んでしまう
        class_id_names = _load_label_map(label_map_path)
        class_names = [name for _, name in sorted(class_id_names)]
    # クラス名からIDへ変換するdict
    class_name_to_id = {n: i for i, n in enumerate(class_names)}

    # IDリストの読み込み
    ids = [
        line
        for line in (data_dir / f"ImageSets/Main/{split}.txt").read_text().split("\n")
        if len(line) > 0
    ]

    # xmlの読み込み
    labels = []
    for id_ in ids:
        anno = ET.parse(str(data_dir / f"Annotations/{id_}.xml"))

        size_ann = anno.find("size")
        width = int(size_ann.find("width").text)  # type: ignore
        height = int(size_ann.find("height").text)  # type: ignore

        bboxes = []
        classes = []
        difficults = []
        for obj in anno.findall("object"):
            assert obj is not None
            if not use_difficult and int(obj.find("difficult").text) == 1:  # type: ignore
                continue

            name = obj.find("name").text.lower().strip()  # type: ignore
            classes.append(class_name_to_id[name])
            bndbox_ann = obj.find("bndbox")
            bboxes.append(
                [
                    int(bndbox_ann.find(tag).text) - 1  # type: ignore
                    for tag in ("xmin", "ymin", "xmax", "ymax")
                ]
            )
            difficults.append(int(obj.find("difficult").text))  # type: ignore

        bboxes = np.array(bboxes, dtype=np.float32)
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] /= width
            bboxes[:, [1, 3]] /= height

        label = tk.od.ObjectsAnnotation(
            path=data_dir / f"JPEGImages/{id_}.jpg",
            width=width,
            height=height,
            classes=classes,
            bboxes=bboxes,
            difficults=difficults,
        )
        labels.append(label)

    return tk.od.ObjectsAnnotation.create_dataset(labels, class_names=class_names)


def _load_label_map(label_map_path):
    """pascal_label_map.pbtxtを読み込む。"""
    class_id_names = []
    id_, name = None, None
    for line in label_map_path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line.startswith("id:"):
            id_ = int(line[3:])
        elif line.startswith("name:"):
            name = line[5:].strip()
            if name[0] == "'" and name[-1] == "'":
                name = (
                    name[1:-1]
                    .encode("ascii", "backslashreplace")
                    .decode("unicode-escape")
                )
        if id_ is not None and name is not None:
            class_id_names.append((id_, name))
            id_, name = None, None
    return class_id_names
