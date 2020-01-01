"""PASCAL VOCデータセット関連。

以下の3ファイルを解凍して出来たVOCdevkitディレクトリのパスを受け取って色々処理をする。

- `VOCtrainval_06-Nov-2007.tar <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar>`_
- `VOCtrainval_11-May-2012.tar <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>`_
- `VOCtest_06-Nov-2007.tar <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>`_

"""
from __future__ import annotations

import pathlib
import typing
import xml.etree.ElementTree

import numpy as np

import pytoolkit as tk

from .. import data as tk_data

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

# 0～19のIDへの変換
CLASS_NAMES_TO_ID = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}


def load_od(vocdevkit_dir):
    """物体検出のデータを読み込む。

    Args:
        vocdevkit_dir: VOCdevkitディレクトリのパス

    Returns:
        - (X_train, y_train): 訓練データ
        - (X_val, y_val): 検証データ
        - class_names: クラス名の配列

    """
    X_train, y_train = load_0712_trainval(vocdevkit_dir)
    X_val, y_val = load_07_test(vocdevkit_dir)
    return (X_train, y_train), (X_val, y_val), CLASS_NAMES


def load_0712_trainval(vocdevkit_dir, class_name_to_id=None):
    """PASCAL VOCデータセットの、07+12 trainvalの読み込み。

    Args:
        vocdevkit_dir: VOCdevkitディレクトリのパス
        class_name_to_id: クラス名からIDへの変換の辞書。Noneなら0～19に変換。

    Returns:
        - X: 画像ファイルのパスのndarray
        - y: tk.od.ObjectsAnnotationのndarray

    """
    X1, y1 = load_set(
        vocdevkit_dir, 2007, "trainval", class_name_to_id, without_difficult=True
    )
    X2, y2 = load_set(
        vocdevkit_dir, 2012, "trainval", class_name_to_id, without_difficult=True
    )
    return np.concatenate([X1, X2]), np.concatenate([y1, y2])


def load_07_test(vocdevkit_dir, class_name_to_id=None):
    """PASCAL VOCデータセットの、07 testの読み込み。

    Args:
        vocdevkit_dir: VOCdevkitディレクトリのパス
        class_name_to_id: クラス名からIDへの変換の辞書。Noneなら0～19に変換。

    Returns:
        - X: 画像ファイルのパスのndarray
        - y: tk.od.ObjectsAnnotationのndarray

    """
    return load_set(vocdevkit_dir, 2007, "test", class_name_to_id)


def load_set(
    vocdevkit_dir, year, set_name, class_name_to_id=None, without_difficult=False
):
    """PASCAL VOCデータセットの読み込み。

    Args:
        vocdevkit_dir: VOCdevkitディレクトリのパス
        year: 2007とか2012とか
        set_name: 'trainval'とか'test'とか
        class_name_to_id: クラス名からIDへの変換の辞書。Noneなら0～19に変換。
        without_difficult: difficultフラグが'1'のものを読み込まないならTrue。

    Returns:
        - X: 画像ファイルのパスのndarray
        - y: tk.od.ObjectsAnnotationのndarray

    """
    vocdevkit_dir = pathlib.Path(vocdevkit_dir)
    names = (
        (vocdevkit_dir / f"VOC{year}" / "ImageSets" / "Main" / f"{set_name}.txt")
        .read_text()
        .split("\n")
    )
    return load_annotations(
        vocdevkit_dir,
        vocdevkit_dir / f"VOC{year}" / "Annotations",
        names,
        class_name_to_id,
        without_difficult,
    )


def load_annotations(
    vocdevkit_dir,
    annotations_dir,
    names=None,
    class_name_to_id=None,
    without_difficult=False,
):
    """VOC2007などのアノテーションデータの読み込み。

    Returns:
        - X: 画像ファイルのパスのndarray
        - y: tk.od.ObjectsAnnotationのndarray
        - names: 「画像ファイル名拡張子なし」のリスト。またはNone。

    """
    d = pathlib.Path(annotations_dir)
    if names is None:
        names = [p.stem for p in d.glob("*.xml")]
    y = [
        load_annotation(
            vocdevkit_dir, d / (name + ".xml"), class_name_to_id, without_difficult
        )
        for name in names
    ]
    return np.array([t.path for t in y]), np.array(y)


def load_annotation(
    vocdevkit_dir, xml_path, class_name_to_id=None, without_difficult=False
):
    """VOC2007などのアノテーションデータの読み込み。"""
    vocdevkit_dir = pathlib.Path(vocdevkit_dir)
    class_name_to_id = class_name_to_id or CLASS_NAMES_TO_ID
    root = xml.etree.ElementTree.parse(str(xml_path)).getroot()
    folder = root.find("folder").text  # type: ignore
    filename = root.find("filename").text  # type: ignore
    size_tree = root.find("size")
    width = float(size_tree.find("width").text)  # type: ignore
    height = float(size_tree.find("height").text)  # type: ignore
    classes = []
    bboxes = []
    difficults = []
    for object_tree in root.findall("object"):
        difficult = object_tree.find("difficult").text == "1"  # type: ignore
        if without_difficult and difficult:
            continue
        class_id = class_name_to_id[object_tree.find("name").text]  # type: ignore
        bndbox = object_tree.find("bndbox")
        xmin = float(bndbox.find("xmin").text) / width  # type: ignore
        ymin = float(bndbox.find("ymin").text) / height  # type: ignore
        xmax = float(bndbox.find("xmax").text) / width  # type: ignore
        ymax = float(bndbox.find("ymax").text) / height  # type: ignore
        classes.append(class_id)
        bboxes.append([xmin, ymin, xmax, ymax])
        difficults.append(difficult)
    annotation = tk.od.ObjectsAnnotation(
        path=vocdevkit_dir / folder / "JPEGImages" / filename,
        width=width,
        height=height,
        classes=classes,
        bboxes=bboxes,
        difficults=difficults,
    )
    return annotation


def evaluate(y_true, y_pred):
    """PASCAL VOC向けのスコア算出。

    ChainerCVを利用。
    https://chainercv.readthedocs.io/en/stable/reference/evaluations.html?highlight=eval_detection_coco#eval-detection-voc

    Returns:
        - 'mAP': 普通っぽい(?)metric。
        - 'mAP_VOC': PASCAL VOC 2007版metric。11点での平均。

    """
    import chainercv

    gt_classes_list = np.array([y.classes for y in y_true])
    gt_bboxes_list = np.array([y.real_bboxes for y in y_true])
    gt_difficults_list = np.array([y.difficults for y in y_true])
    pred_classes_list = np.array([p.classes for p in y_pred])
    pred_confs_list = np.array([p.confs for p in y_pred])
    pred_bboxes_list = np.array(
        [p.get_real_bboxes(y.width, y.height) for (p, y) in zip(y_pred, y_true)]
    )
    scores1 = chainercv.evaluations.eval_detection_voc(
        pred_bboxes_list,
        pred_classes_list,
        pred_confs_list,
        gt_bboxes_list,
        gt_classes_list,
        gt_difficults_list,
        use_07_metric=False,
    )
    scores2 = chainercv.evaluations.eval_detection_voc(
        pred_bboxes_list,
        pred_classes_list,
        pred_confs_list,
        gt_bboxes_list,
        gt_classes_list,
        gt_difficults_list,
        use_07_metric=True,
    )
    return {
        "AP": scores1["ap"],
        "mAP": scores1["map"],
        "AP_VOC": scores2["ap"],
        "mAP_VOC": scores2["map"],
    }


def load_voc_od(voc_dir, use_crowded=False):
    """VOCの物体検出のデータを読み込む。

    References:
        - <https://chainercv.readthedocs.io/en/stable/reference/datasets.html#vocbboxdataset>

    """
    from chainercv.datasets.voc.voc_bbox_dataset import VOCBboxDataset

    ds_train07 = VOCBboxDataset(
        data_dir=str(voc_dir), split="trainval", year="2007", use_difficult=False,
    )
    # TODO
    # ds_train12 = VOCBboxDataset(
    #     data_dir=str(voc_dir), split="trainval", year="2012", use_difficult=False,
    # )
    ds_val = VOCBboxDataset(
        data_dir=str(voc_dir),
        split="test",
        year="2007",
        use_difficult=True,
        return_difficult=True,
    )

    return _VOCODDataset(ds_train07), _VOCODDataset(ds_val)


class _VOCODDataset(tk_data.Dataset):
    """chainercvのDatasetからtk.data.Datasetへの変換"""

    def __init__(self, ds):
        self.ds = ds
        super().__init__(data=np.arange(len(self.ds)))

    def get_data(self, index: int) -> typing.Tuple[typing.Any, typing.Any]:
        img, bboxes, classes, areas, crowdeds = self.ds[index]
        X = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # CHW -> HWC
        assert X.shape[-1] == 3, f"shape error: {X.shape}"  # RGB

        # (ymin,xmin,ymax,xmax) -> (xmin,ymin,xmax,ymax)
        bboxes = bboxes[:, [1, 0, 3, 2]].astype(np.float32)
        bboxes[:, [0, 2]] /= X.shape[1]
        bboxes[:, [1, 3]] /= X.shape[0]

        y = tk.od.ObjectsAnnotation(
            path=self._get_path(index),
            width=X.shape[1],
            height=X.shape[0],
            classes=classes,
            bboxes=bboxes,
            areas=areas,
            crowdeds=crowdeds,
        )

        return X, y

    def _get_path(self, index: int) -> pathlib.Path:
        # https://github.com/chainer/chainercv/blob/fddc813/chainercv/datasets/voc/voc_bbox_dataset.py#L84
        return (
            pathlib.Path(self.ds.data_dir) / "JPEGImages" / f"{self.ds.ids[index]}.jpg"
        )
