"""MS COCOデータセット関連。

以下の3ファイルを解凍した結果を格納しているディレクトリのパスを受け取って色々処理をする。

- [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
- [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

"""
from __future__ import annotations

import pathlib
import typing

import numpy as np

import pytoolkit as tk


def load_coco_od(
    coco_dir: tk.typing.PathLike, use_crowded: bool = False, year: int = 2017
) -> typing.Tuple[tk.data.Dataset, tk.data.Dataset]:
    """COCOの物体検出のデータを読み込む。"""
    ds_train = load_od_data(coco_dir, f"train{year}", use_crowded=use_crowded)
    ds_val = load_od_data(coco_dir, f"val{year}", use_crowded=True)
    assert tuple(ds_train.metadata["class_names"]) == tuple(
        ds_val.metadata["class_names"]
    )
    return ds_train, ds_val


def load_od_data(coco_dir, data_name, use_crowded):
    """物体検出のデータの読み込み。"""
    import pycocotools.coco

    coco_dir = pathlib.Path(coco_dir)
    coco = pycocotools.coco.COCO(
        str(coco_dir / "annotations" / f"instances_{data_name}.json")
    )

    class_names = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    jsonclassid_to_index = {
        c["id"]: class_names.index(c["name"]) for c in coco.loadCats(coco.getCatIds())
    }

    labels = []
    for entry in coco.loadImgs(coco.getImgIds()):
        dirname, filename = entry["coco_url"].split("/")[-2:]
        objs = coco.loadAnns(
            coco.getAnnIds(imgIds=entry["id"], iscrowd=None if use_crowded else False)
        )

        bboxes, classes, areas, crowdeds = [], [], [], []
        width, height = entry["width"], entry["height"]
        for obj in objs:
            if obj.get("ignore", 0) == 1:
                continue
            x, y, w, h = obj["bbox"]
            bbox = np.array([x, y, x + w, y + h]) / np.array(
                [width, height, width, height]
            )
            bbox = np.clip(bbox, 0, 1)
            if (bbox[:2] < bbox[2:]).all():
                bboxes.append(bbox)
                classes.append(jsonclassid_to_index[obj["category_id"]])
                areas.append(obj["area"])
                crowdeds.append(obj["iscrowd"])

        labels.append(
            tk.od.ObjectsAnnotation(
                path=coco_dir / dirname / filename,
                width=width,
                height=height,
                classes=classes,
                bboxes=bboxes,
                areas=areas,
                crowdeds=crowdeds,
            )
        )
    return tk.od.ObjectsAnnotation.create_dataset(labels, class_names=class_names)
