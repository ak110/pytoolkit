"""MS COCOデータセット関連。

以下の3ファイルを解凍した結果を格納しているディレクトリのパスを受け取って色々処理をする。

- [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
- [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

"""
import pathlib
import typing

import numpy as np

import pytoolkit as tk


def load_coco_od(coco_dir, use_crowded=False):
    """MSCOCOの物体検出のデータを読み込む。"""
    from chainercv.datasets.coco.coco_bbox_dataset import COCOBboxDataset

    ds_train = COCOBboxDataset(
        data_dir=str(coco_dir),
        split="train",
        year="2017",
        use_crowded=use_crowded,
        return_area=True,
        return_crowded=True,
    )
    ds_val = COCOBboxDataset(
        data_dir=str(coco_dir),
        split="val",
        year="2017",
        use_crowded=use_crowded,
        return_area=True,
        return_crowded=True,
    )

    class COCOODDataset(tk.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
            super().__init__(data=np.arange(len(self.ds)))

        def get_data(self, index: int) -> typing.Tuple[typing.Any, typing.Any]:
            img, bboxes, classes, areas, crowdeds = self.ds[index]
            y = tk.od.ObjectsAnnotation(
                path=None,
                width=img.shape[1],
                height=img.shape[0],
                classes=classes,
                bboxes=bboxes,
                areas=areas,
                crowdeds=crowdeds,
            )
            return img, y

    return COCOODDataset(ds_train), COCOODDataset(ds_val)


def load_od(coco_dir, year=2017):
    """物体検出のデータを読み込む。

    Args:
        coco_dir: annotationsディレクトリなどが入っているディレクトリのパス
        year: 読み込むデータの西暦

    Returns:
        - (X_train, y_train)
        - (X_val, y_val)
        - class_names

    """
    X_train, y_train, class_names = load_od_data(coco_dir, f"train{year}")
    X_val, y_val, class_names_val = load_od_data(coco_dir, f"val{year}")
    assert class_names == class_names_val
    return (X_train, y_train), (X_val, y_val), class_names


def load_od_data(coco_dir, data_name):
    """物体検出のデータの読み込み。"""
    from pycocotools import coco

    coco_dir = pathlib.Path(coco_dir)
    coco = coco.COCO(str(coco_dir / "annotations" / f"instances_{data_name}.json"))

    class_names = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    jsonclassid_to_index = {
        c["id"]: class_names.index(c["name"]) for c in coco.loadCats(coco.getCatIds())
    }

    annotations = []
    for entry in coco.loadImgs(coco.getImgIds()):
        dirname, filename = entry["coco_url"].split("/")[-2:]
        objs = coco.loadAnns(coco.getAnnIds(imgIds=entry["id"], iscrowd=None))

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

        annotations.append(
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

    return np.array([t.path for t in annotations]), np.array(annotations), class_names


def load_ss(coco_dir, cache_dir, input_size=None, year=2017):
    """セマンティックセグメンテーションのデータを読み込む。

    Args:
        coco_dir: annotationsディレクトリなどが入っているディレクトリのパス
        year: 読み込むデータの西暦
        cache_dir: キャッシュの保存先ディレクトリ
        input_size: 読み込み時にリサイズする場合、そのサイズ

    Returns:
        - (X_train, y_train): 訓練データ
        - (X_val, y_val): 検証データ
        - class_names: クラス名の配列

    """
    X_train, y_train, class_names = load_ss_data(
        coco_dir, f"train{year}", cache_dir, input_size
    )
    X_val, y_val, class_names_val = load_ss_data(
        coco_dir, f"val{year}", cache_dir, input_size
    )
    assert class_names == class_names_val
    return (X_train, y_train), (X_val, y_val), class_names


def load_ss_data(coco_dir, data_name, cache_dir, input_size=None):
    """セマンティックセグメンテーションのデータの読み込み。"""
    from pycocotools import coco, mask as cocomask

    coco_dir = pathlib.Path(coco_dir)
    cache_dir = pathlib.Path(cache_dir)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    coco = coco.COCO(str(coco_dir / "annotations" / f"instances_{data_name}.json"))

    class_names = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    jsonclassid_to_index = {
        c["id"]: class_names.index(c["name"]) for c in coco.loadCats(coco.getCatIds())
    }

    X, y = [], []
    for entry in tk.utils.tqdm(coco.loadImgs(coco.getImgIds()), desc="load_ss_data"):
        dirname, filename = entry["coco_url"].split("/")[-2:]
        save_path = cache_dir / dirname / (filename + ".npy")
        X.append(coco_dir / dirname / filename)
        y.append(save_path)
        if not save_path.exists():
            # 読み込み
            objs = coco.loadAnns(coco.getAnnIds(imgIds=entry["id"], iscrowd=None))
            mask = np.zeros(
                (entry["height"], entry["width"], len(class_names)), dtype=np.uint8
            )
            for obj in objs:
                if obj.get("ignore", 0) == 1:
                    continue
                rle = cocomask.frPyObjects(
                    obj["segmentation"], entry["height"], entry["width"]
                )
                m = cocomask.decode(rle)
                class_id = jsonclassid_to_index[obj["category_id"]]
                mask[:, :, class_id] |= m
            mask = np.where(mask, np.uint8(255), np.uint8(0))
            # リサイズ
            if input_size is not None:
                mask = tk.ndimage.resize(mask, input_size[1], input_size[0])
            # 保存
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), mask)

    return np.array(X), np.array(y), class_names


def evaluate(y_true, y_pred):
    """MS COCO向けのスコア算出。

    ChainerCVを利用。
    https://chainercv.readthedocs.io/en/stable/reference/evaluations.html?highlight=eval_detection_coco#chainercv.evaluations.eval_detection_coco

    Returns:
        - "map/iou=0.50:0.95/area=all/max_dets=100"
        - "map/iou=0.50/area=all/max_dets=100"
        - "map/iou=0.75/area=all/max_dets=100"
        - などなど

    """
    import chainercv

    gt_classes_list = np.array([y.classes for y in y_true])
    gt_bboxes_list = np.array([y.real_bboxes for y in y_true])
    gt_areas_list = np.array([y.areas for y in y_true])
    gt_crowdeds_list = np.array([y.crowdeds for y in y_true])
    pred_classes_list = np.array([p.classes for p in y_pred])
    pred_confs_list = np.array([p.confs for p in y_pred])
    pred_bboxes_list = np.array(
        [p.get_real_bboxes(y.width, y.height) for (p, y) in zip(y_pred, y_true)]
    )
    return chainercv.evaluations.eval_detection_coco(
        pred_bboxes_list,
        pred_classes_list,
        pred_confs_list,
        gt_bboxes_list,
        gt_classes_list,
        gt_areas_list,
        gt_crowdeds_list,
    )
