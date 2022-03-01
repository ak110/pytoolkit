"""物体検出関連。"""
from __future__ import annotations

import dataclasses
import pathlib
import typing
import warnings

import cv2
import numpy as np

import pytoolkit as tk


@dataclasses.dataclass
class ObjectsAnnotation:
    """物体検出のアノテーションデータを持つためのクラス。

    Args:
        path: 画像ファイルのパス
        width: 画像の横幅[px]
        height: 画像の縦幅[px]
        classes: クラスIDのndarray。値は[0, num_classes)の整数。shapeは(物体数,)
        bboxes: bounding box(x1, y1, x2, y2)のndarray。値は[0, 1]。shapeは(物体数, 4)
        difficults: difficultフラグ(PASCAL VOCデータセット等で使用)のndarray。値はTrue or False。shapeは(物体数,)
        areas: 面積 (MS COCO用)
        crowdeds: クラウドソーシングでアノテーションされたか否か (MS COCO用)

    """

    path: pathlib.Path
    width: int
    height: int
    classes: np.ndarray
    bboxes: np.ndarray
    difficults: np.ndarray | None = None
    areas: np.ndarray | None = None
    crowdeds: np.ndarray | None = None

    @staticmethod
    def create_dataset(
        labels: typing.Sequence[ObjectsAnnotation], class_names: list[str] = None
    ) -> tk.data.Dataset:
        """ObjectsAnnotationの配列からDatasetを作成する。

        Args:
            labels: ObjectsAnnotationの配列
            class_names: クラス名の配列

        Return:
            Dataset

        """
        data = np.array([y.path for y in labels])
        ds = tk.data.Dataset(data=data, labels=np.asarray(labels))
        if class_names is not None:
            ds.metadata["class_names"] = class_names
        return ds

    def __init__(
        self,
        path,
        width,
        height,
        classes,
        bboxes,
        difficults=None,
        areas=None,
        crowdeds=None,
    ):
        self.path = pathlib.Path(path)
        self.width = width
        self.height = height
        self.classes = np.asarray(classes, dtype=np.int32)
        self.bboxes = np.asarray(bboxes, dtype=np.float32)
        self.difficults = (
            np.asarray(difficults, dtype=np.bool_) if difficults is not None else None
        )
        self.areas = np.asarray(areas, dtype=np.float32) if areas is not None else None
        self.crowdeds = (
            np.asarray(crowdeds, dtype=np.bool_) if crowdeds is not None else None
        )
        self.__post_init__()

    def __post_init__(self):
        assert len(self.classes) == len(self.bboxes)
        assert self.width >= 1, f"Value error: {self.width=}"
        assert self.height >= 1, f"Value error: {self.height=}"
        assert self.classes.dtype == np.int32, f"dtype error: {self.classes.dtype}"
        assert self.bboxes.dtype == np.float32, f"dtype error: {self.bboxes.dtype}"
        if self.num_objects == 0:
            self.bboxes = self.bboxes.reshape((self.num_objects, 4))
        assert self.classes.shape == (
            self.num_objects,
        ), f"Shape error: {self.classes.shape}"
        assert self.bboxes.shape == (
            self.num_objects,
            4,
        ), f"Shape error: {self.bboxes.shape}"
        assert (self.bboxes >= 0).all(), f"Value error: {self.bboxes}"
        assert (self.bboxes <= 1).all(), f"Value error: {self.bboxes}"
        assert (
            self.bboxes[:, :2] < self.bboxes[:, 2:]
        ).all(), f"Value error: {self.bboxes}"
        if self.difficults is not None:
            assert (
                self.difficults.dtype == np.bool_
            ), f"dtype error: {self.difficults.dtype}"
            assert self.difficults.shape == (
                self.num_objects,
            ), f"Shape error: {self.difficults.shape}"
        if self.areas is not None:
            assert self.areas.dtype == np.float32, f"dtype error: {self.areas.dtype}"
            assert self.areas.shape == (
                self.num_objects,
            ), f"Shape error: {self.areas.shape=}"
        if self.crowdeds is not None:
            assert (
                self.crowdeds.dtype == np.bool_
            ), f"dtype error: {self.crowdeds.dtype}"
            assert self.crowdeds.shape == (
                self.num_objects,
            ), f"Shape error: {self.crowdeds.shape=}"

    @property
    def num_objects(self):
        """物体の数を返す。"""
        return len(self.classes)

    @property
    def real_bboxes(self):
        """実ピクセル数換算のbboxesを返す。"""
        return np.round(
            self.bboxes * [self.width, self.height, self.width, self.height]
        ).astype(np.int32)

    @property
    def bboxes_ar_fixed(self):
        """縦横比を補正したbboxesを返す。(prior box算出など用)"""
        assert self.width > 0 and self.height > 0
        bboxes = np.copy(self.bboxes)
        if self.width > self.height:
            # 横長の場合、上下にパディングする想定で補正
            bboxes[:, [1, 3]] *= self.height / self.width
        elif self.width < self.height:
            # 縦長の場合、左右にパディングする想定で補正
            bboxes[:, [0, 2]] *= self.width / self.height
        return bboxes

    def to_str(self, class_names):
        """表示用の文字列化"""
        a = [
            f"({x1}, {y1}) [{x2 - x1} x {y2 - y1}]: {class_names[c]}"
            for (x1, y1, x2, y2), c in sorted(
                zip(self.real_bboxes, self.classes), key=lambda x: rbb_sortkey(x[0])
            )
        ]
        return "\n".join(a)

    def plot(self, img, class_names, conf_threshold=0, max_long_side=None):
        """ワクを描画した画像を作って返す。"""
        return plot_objects(
            img,
            self.classes,
            None,
            self.bboxes,
            class_names,
            conf_threshold=conf_threshold,
            max_long_side=max_long_side,
        )

    def rot90(self, k):
        """90度回転。"""
        assert 0 <= k <= 3
        if k == 1:
            self.bboxes = self.bboxes[:, [1, 0, 3, 2]]
            self.bboxes[:, [1, 3]] = 1 - self.bboxes[:, [3, 1]]
        elif k == 2:
            self.bboxes = 1 - self.bboxes[:, [2, 3, 0, 1]]
        elif k == 3:
            self.bboxes = self.bboxes[:, [1, 0, 3, 2]]
            self.bboxes[:, [0, 2]] = 1 - self.bboxes[:, [2, 0]]


@dataclasses.dataclass
class ObjectsPrediction:
    """物体検出の予測結果を持つクラス。

    Args:
        classes: クラスIDのndarray。値は[0, num_classes)の整数。shapeは(物体数,)
        confs: 確信度のndarray。値は[0, 1]。shapeは(物体数,)
        bboxes: bounding box(x1, y1, x2, y2)のndarray。値は[0, 1]。shapeは(物体数, 4)

    """

    classes: np.ndarray
    confs: np.ndarray
    bboxes: np.ndarray

    def __init__(self, classes, confs, bboxes):
        self.classes = np.asarray(classes, dtype=np.int32)
        self.confs = np.asarray(confs, dtype=np.float32)
        self.bboxes = np.asarray(bboxes, dtype=np.float32)
        self.__post_init__()

    def __post_init__(self):
        assert self.classes.shape == (self.num_objects,)
        assert self.confs.shape == (self.num_objects,)
        assert self.bboxes.shape == (self.num_objects, 4)

    def apply_threshold(self, conf_threshold: float) -> ObjectsPrediction:
        """確信度が閾値未満の物体を削除したものを作成して返す。"""
        mask = self.confs >= conf_threshold
        return ObjectsPrediction(
            classes=self.classes[mask], confs=self.confs[mask], bboxes=self.bboxes[mask]
        )

    @property
    def num_objects(self):
        """物体の数を返す。"""
        return len(self.classes)

    def to_str(self, width, height, class_names, conf_threshold=0):
        """表示用の文字列化"""
        a = [
            f"({x1}, {y1}) [{x2 - x1} x {y2 - y1}]: {class_names[c]}"
            for (x1, y1, x2, y2), c, cf in sorted(
                zip(self.get_real_bboxes(width, height), self.classes, self.confs),
                key=lambda x: rbb_sortkey(x[0]),
            )
            if cf >= conf_threshold
        ]
        return "\n".join(a)

    def plot(self, img, class_names=None, conf_threshold=0, max_long_side=None):
        """ワクを描画した画像を作って返す。"""
        return plot_objects(
            img,
            self.classes,
            self.confs,
            self.bboxes,
            class_names,
            conf_threshold=conf_threshold,
            max_long_side=max_long_side,
        )

    def is_match(self, classes, bboxes, conf_threshold=0, iou_threshold=0.5):
        """classes/bboxesと過不足なく一致していたらTrueを返す。"""
        assert bboxes.shape == (len(classes), 4)

        mask = self.confs >= conf_threshold
        if np.sum(mask) != len(classes):
            return False  # 数が不一致
        if len(classes) == 0:
            return True  # OK

        iou = compute_iou(bboxes, self.bboxes[mask])
        iou_max = iou.max(axis=0)
        if (iou_max < iou_threshold).any():
            return False  # IoUが閾値未満

        pred_gt = iou.argmax(axis=0)
        if len(np.unique(pred_gt)) != len(classes):
            return False  # 1:1になっていない

        for gt_ix, gt_class in enumerate(classes):
            if gt_class != self.classes[mask][pred_gt == gt_ix][0]:
                return False  # クラスが不一致

        return True  # OK

    def get_real_bboxes(self, width, height):
        """実ピクセル数換算のbboxesを返す。"""
        return np.round(self.bboxes * [width, height, width, height]).astype(np.int32)

    def crop(self, img, conf_threshold=0):
        """Bounding boxで切り出した画像を返す。"""
        img = tk.ndimage.load(img, grayscale=False)
        height, width = img.shape[:2]
        return [
            tk.ndimage.crop(img, x1, y1, x2 - x1, y2 - y1)
            for (x1, y1, x2, y2), cf in zip(
                self.get_real_bboxes(width, height), self.confs
            )
            if cf >= conf_threshold
        ]


def search_conf_threshold(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
    iou_threshold: float = 0.5,
):
    """物体検出の正解と予測結果から、F1スコアが最大になるconf_thresholdを返す。"""
    conf_threshold_list = np.linspace(0.01, 0.99, 50)
    scores = []
    for conf_th in conf_threshold_list:
        _, _, fscores, supports = compute_scores(y_true, y_pred, conf_th, iou_threshold)
        score = np.average(fscores, weights=supports)  # sklearnで言うaverage='weighted'
        scores.append(score)
    scores = np.array(scores)
    max_scores = scores >= 1
    if max_scores.any():  # 満点が1つ以上存在する場合、そのときの閾値の平均を返す(怪)
        return np.mean(conf_threshold_list[max_scores])
    return conf_threshold_list[scores.argmax()]


def od_accuracy(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
    conf_threshold: float = 0.0,
    iou_threshold: float = 0.5,
):
    """物体検出で過不足なく検出できた時だけ正解扱いとした正解率を算出する。"""
    assert len(y_true) == len(y_pred)
    assert 0 < iou_threshold < 1
    assert 0 <= conf_threshold < 1
    return np.mean(
        [
            yp.is_match(yt.classes, yt.bboxes, conf_threshold, iou_threshold)
            for yt, yp in zip(y_true, y_pred)
        ]
    )


def compute_scores(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
    conf_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    num_classes: int = None,
):
    """物体検出の正解と予測結果から、適合率、再現率、F値、該当回数を算出して返す。"""
    assert len(y_true) == len(y_pred)
    assert 0 < iou_threshold < 1
    assert 0 <= conf_threshold < 1
    if num_classes is None:
        num_classes = np.max(np.concatenate([y.classes for y in y_true])) + 1

    tp = np.zeros((num_classes,), dtype=np.int32)  # true positive
    fp = np.zeros((num_classes,), dtype=np.int32)  # false positive
    fn = np.zeros((num_classes,), dtype=np.int32)  # false negative

    for yt, yp in zip(y_true, y_pred):
        # conf_threshold以上をいったんすべて対象とする
        pred_enabled = yp.confs >= conf_threshold
        # 各正解が予測結果に含まれるか否か: true positive/negative
        difficults = (
            yt.difficults
            if yt.difficults is not None
            else np.zeros((yt.num_objects,), dtype=np.bool_)
        )
        for gt_class, gt_bbox, gt_difficult in zip(yt.classes, yt.bboxes, difficults):
            pred_mask = np.logical_and(pred_enabled, yp.classes == gt_class)
            if pred_mask.any():
                pred_bboxes = yp.bboxes[pred_mask]
                iou = compute_iou(np.expand_dims(gt_bbox, axis=0), pred_bboxes)[0, :]
                pred_ix = iou.argmax()
                pred_iou = iou[pred_ix]
            else:
                pred_ix = None
                pred_iou = -1  # 検出失敗
            if pred_iou >= iou_threshold:
                # 検出成功
                if not gt_difficult:
                    tp[gt_class] += 1
                assert pred_ix is not None
                pred_enabled[np.where(pred_mask)[0][pred_ix]] = False
            else:
                # 検出失敗
                if not gt_difficult:
                    fn[gt_class] += 1
        # 正解に含まれなかった予測結果: false positive
        for pred_class in yp.classes[pred_enabled]:
            fp[pred_class] += 1

    supports = tp + fn
    precisions = tp.astype(float) / (tp + fp + 1e-7)
    recalls = tp.astype(float) / (supports + 1e-7)
    fscores = 2 / (1 / (precisions + 1e-7) + 1 / (recalls + 1e-7))
    return precisions, recalls, fscores, supports


def confusion_matrix(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
    conf_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    num_classes: int = None,
):
    """物体検出用の混同行列を作る。

    分類と異なり、検出漏れと誤検出があるのでその分列と行を1つずつ増やしたものを返す。
    difficultは扱いが難しいので無視。
    """
    assert len(y_true) == len(y_pred)
    assert 0 < iou_threshold < 1
    assert 0 <= conf_threshold < 1
    if num_classes is None:
        num_classes = np.max(np.concatenate([y.classes for y in y_true])) + 1

    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    for yt, yp in zip(y_true, y_pred):
        pred_enabled = yp.confs >= conf_threshold
        if yt.num_objects > 0:
            if yp.num_objects > 0:
                iou = compute_iou(yt.bboxes, yp.bboxes)
                pred_gt = iou.argmax(axis=0)  # 一番近いboxにマッチさせる (やや怪しい)
                pred_iou_mask = iou.max(axis=0) >= iou_threshold
                # 正解毎にループ
                for gt_ix, gt_class in enumerate(yt.classes):
                    m = np.logical_and(
                        pred_enabled, pred_iou_mask
                    )  # まだ使用済みでなく、IoUが大きく、
                    m = np.logical_and(m, pred_gt == gt_ix)  # IoUの対象がgt_ixなものが対象
                    pred_targets = yp.confs.argsort()[::-1][m]  # 対象のindexを確信度順で
                    found = False
                    # クラスもあってる検出
                    for pred_ix in pred_targets:
                        pc = yp.classes[pred_ix]
                        if gt_class == pc:
                            if found:
                                cm[-1, pc] += 1  # 誤検出(重複)
                            else:
                                found = True
                                cm[gt_class, pc] += 1  # 検出成功
                            # 一度カウントしたものは次から無視
                            pred_enabled[pred_ix] = False
                    # クラス違い
                    for pred_ix in pred_targets:
                        pc = yp.classes[pred_ix]
                        if pred_enabled[pred_ix] and gt_class != pc:
                            if found:
                                cm[-1, pc] += 1  # 誤検出(重複&クラス違い)
                            else:
                                # ここでFound=Trueは微妙だが、gt_classの数が合わなくなるので1個だけにする。。
                                found = True
                                cm[gt_class, pc] += 1  # 誤検出(クラス違い)
                        # 一度カウントしたものは次から無視
                        pred_enabled[pred_ix] = False
                    if not found:
                        cm[gt_class, -1] += 1  # 検出漏れ
            else:
                # 全て検出漏れ
                for gt_class in yt.classes:
                    cm[gt_class, -1] += 1  # 検出漏れ
        # 余った予測結果：誤検出
        for pred_class in yp.classes[pred_enabled]:
            cm[-1, pred_class] += 1

    return cm


def compute_iou(bboxes_a, bboxes_b):
    """IoU(Intersection over union、Jaccard係数)の算出。重なり具合を示す係数。(0～1)"""
    assert bboxes_a.shape[0] > 0
    assert bboxes_b.shape[0] > 0
    assert bboxes_a.shape == (len(bboxes_a), 4)
    assert bboxes_b.shape == (len(bboxes_b), 4)
    lt = np.maximum(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2])
    rb = np.minimum(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    area_inter = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    area_union = area_a[:, np.newaxis] + area_b - area_inter
    iou = area_inter / area_union
    assert iou.shape == (len(bboxes_a), len(bboxes_b))
    return iou


def is_intersection(bboxes_a, bboxes_b):
    """boxes_aとboxes_bでそれぞれ交差している部分が存在するか否かを返す。"""
    assert bboxes_a.shape[0] > 0
    assert bboxes_b.shape[0] > 0
    assert bboxes_a.shape == (len(bboxes_a), 4)
    assert bboxes_b.shape == (len(bboxes_b), 4)
    lt = np.maximum(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2])
    rb = np.minimum(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    return (lt < rb).all(axis=-1)


def is_in_box(boxes_a, boxes_b):
    """boxes_aがboxes_bの中に完全に入っているならtrue。"""
    assert boxes_a.shape == (len(boxes_a), 4)
    assert boxes_b.shape == (len(boxes_b), 4)
    lt = boxes_a[:, np.newaxis, :2] >= boxes_b[:, :2]
    rb = boxes_a[:, np.newaxis, 2:] <= boxes_b[:, 2:]
    return np.logical_and(lt, rb).all(axis=-1)


def plot_objects(
    base_image: np.ndarray,
    classes: np.ndarray | None,
    confs: np.ndarray | None,
    bboxes: np.ndarray,
    class_names: typing.Sequence[str] = None,
    conf_threshold: float = 0.0,
    max_long_side: int = None,
):
    """画像＋オブジェクト([class_id + confidence + xmin/ymin/xmax/ymax]×n)を画像化する。

    Args:
        base_image: 元画像ファイルのパスまたはndarray
        max_long_side: 長辺の最大長(ピクセル数)。超えていたら縮小する。
        classes: クラスIDのリスト
        confs: confidenceのリスト (None可)
        bboxes: xmin/ymin/xmax/ymaxのリスト (それぞれ0.0 ～ 1.0)
        class_names: クラスID→クラス名のリスト  (None可)
        conf_threshold: この値以上のオブジェクトのみ描画する

    """
    confs_ = np.full(len(bboxes), None) if confs is None else confs
    classes_ = np.full(len(bboxes), None) if classes is None else classes
    assert len(confs_) == len(bboxes)
    assert len(classes_) == len(bboxes)
    if class_names is not None and classes is not None:
        assert 0 <= np.min(classes, initial=0) < len(class_names)
        assert 0 <= np.max(classes, initial=0) < len(class_names)

    img = tk.ndimage.load(base_image, grayscale=False)
    if max_long_side is not None and max(*img.shape[:2]) > max_long_side:
        r = max_long_side / max(*img.shape[:2])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_AREA,
        )
    if img.ndim == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    num_classes = len(class_names) if class_names is not None else 1

    colors = cv2.applyColorMap(
        np.linspace(0, 255, num_classes + 1, dtype=np.uint8)[:num_classes],
        cv2.COLORMAP_HSV,
    )
    colors = np.squeeze(colors, axis=1)  # (num_classes, 1, 3) → (num_classes, 3)

    # 確信度の昇順に描画 (確信度が高いほど手前に出るように)
    for clazz, conf, bbox in sorted(
        zip(classes, confs, bboxes), key=lambda x: x[1] if x[1] is not None else 0
    ):
        if conf is not None and conf < conf_threshold:
            continue  # skip
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            warnings.warn("Negative size bbox")
        xmin = max(int(round(bbox[0] * img.shape[1])), 0)
        ymin = max(int(round(bbox[1] * img.shape[0])), 0)
        xmax = min(int(round(bbox[2] * img.shape[1])), img.shape[1])
        ymax = min(int(round(bbox[3] * img.shape[0])), img.shape[0])
        if clazz is None:
            color = colors[0]
        else:
            color = colors[clazz % len(colors)]
            label = (
                class_names[clazz] if class_names is not None else f"class{clazz:02d}"
            )
            text = label if conf is None else f"{conf:0.2f}, {label}"
            text_width = 6 * len(text)
            cv2.rectangle(
                img,
                rec=(xmin - 1, ymin, text_width + 25, 15),
                color=color.tolist(),
                thickness=cv2.FILLED,
            )
            luminance = color[2] * 0.298912 + color[1] * 0.586611 + color[0] * 0.114478
            text_color = [0, 0, 0] if luminance >= 127.5 else [255, 255, 255]
            cv2.putText(
                img,
                text,
                org=(xmin + 5, ymin + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35,
                color=text_color,
                thickness=1,
            )
        cv2.rectangle(
            img,
            rec=(xmin, ymin, xmax - xmin, ymax - ymin),
            color=color.tolist(),
            thickness=2,
        )

    return img


def rbb_sortkey(bb):
    """real_bboxesのソートキーを作って返す。"""
    x1, y1, x2, y2 = bb
    return f"{y1:05d}-{x1:05d}-{y2:05d}-{x2:05d}"
