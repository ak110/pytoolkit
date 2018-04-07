"""機械学習(主にsklearn)関連。"""
import collections
import itertools
import json
import multiprocessing as mp
import pathlib
import xml.etree.ElementTree

import numpy as np
import sklearn.base
import sklearn.cluster
import sklearn.externals.joblib as joblib
import sklearn.model_selection
import sklearn.utils

from . import draw

# VOC2007のクラス名のリスト (20クラス)
VOC_CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
# 0～19のIDへの変換
VOC_CLASS_NAMES_TO_ID = {class_name: i for i, class_name in enumerate(VOC_CLASS_NAMES)}


class ObjectsAnnotation(object):
    """VOC2007などのアノテーションデータを持つためのクラス。"""

    def __init__(self, folder, filename, width, height, classes, bboxes, difficults=None):
        assert len(classes) == len(bboxes)
        assert difficults is None or len(classes) == len(difficults)
        # 画像のフォルダパス (データのディレクトリからの相対パス)
        self.folder = folder
        # ファイル名
        self.filename = filename
        # 画像の横幅(px)
        self.width = width
        # 画像の縦幅(px)
        self.height = height
        # クラスIDの配列
        self.classes = np.array(classes)
        # bounding box(x1, y1, x2, y2)の配列。(0～1)
        self.bboxes = np.array(bboxes)
        # difficultフラグの配列。(True or False)
        self.difficults = np.array(difficults) if difficults is not None else np.zeros(len(classes))
        assert self.width >= 1
        assert self.height >= 1
        assert (self.bboxes >= 0).all()
        assert (self.bboxes <= 1).all()
        assert (self.bboxes[:, :2] < self.bboxes[:, 2:]).all()

    @staticmethod
    def get_path_list(data_dir, y):
        """画像ファイルのパス(pathlib.Path)のndarrayを作って返す。

        # 引数
        - data_dir: VOCdevkitディレクトリが置いてあるディレクトリのパス
        - y: ObjectsAnnotationの配列

        """
        data_dir = pathlib.Path(data_dir)
        return np.array([data_dir / y_.folder / y_.filename for y_ in y])

    @classmethod
    def load_voc_0712_trainval(cls, data_dir, class_name_to_id, without_difficult=False):
        """PASCAL VOCデータセットの、よくある07+12 trainvalの読み込み。"""
        y1 = cls.load_voc(data_dir, 2007, 'trainval', class_name_to_id, without_difficult=without_difficult)
        y2 = cls.load_voc(data_dir, 2012, 'trainval', class_name_to_id, without_difficult=without_difficult)
        return np.concatenate([y1, y2])

    @classmethod
    def load_voc_07_test(cls, data_dir, class_name_to_id):
        """PASCAL VOCデータセットの、よくある07 testの読み込み。"""
        return cls.load_voc(data_dir, 2007, 'test', class_name_to_id)

    @classmethod
    def load_voc(cls, data_dir, year, set_name, class_name_to_id, without_difficult=False):
        """PASCAL VOCデータセットの読み込み。

        # 引数
        - data_dir: VOCdevkitディレクトリが置いてあるディレクトリのパス
        - year: 2007とか2012とか
        - set_name: 'trainval'とか'test'とか

        """
        from . import io
        names = io.read_all_lines(data_dir / f'VOCdevkit/VOC{year}/ImageSets/Main/{set_name}.txt')
        y = cls.load_voc_files(data_dir / f'VOCdevkit/VOC{year}/Annotations',
                               names, class_name_to_id, without_difficult)
        return np.array(y)

    @classmethod
    def load_voc_files(cls, annotations_dir, names, class_name_to_id, without_difficult=False):
        """VOC2007などのアノテーションデータの読み込み。

        namesは「画像ファイル名拡張子なし」のリスト。
        戻り値はObjectsAnnotationの配列。
        """
        d = pathlib.Path(annotations_dir)
        return [cls.load_voc_file(d / (name + '.xml'), class_name_to_id, without_difficult)
                for name in names]

    @staticmethod
    def load_voc_file(f, class_name_to_id, without_difficult):
        """VOC2007などのアノテーションデータの読み込み。"""
        root = xml.etree.ElementTree.parse(str(f)).getroot()
        folder = root.find('folder').text
        filename = root.find('filename').text
        size_tree = root.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        classes = []
        bboxes = []
        difficults = []
        for object_tree in root.findall('object'):
            difficult = object_tree.find('difficult').text == '1'
            if without_difficult and difficult:
                continue
            class_id = class_name_to_id[object_tree.find('name').text]
            bndbox = object_tree.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / width
            ymin = float(bndbox.find('ymin').text) / height
            xmax = float(bndbox.find('xmax').text) / width
            ymax = float(bndbox.find('ymax').text) / height
            classes.append(class_id)
            bboxes.append([xmin, ymin, xmax, ymax])
            difficults.append(difficult)
        annotation = ObjectsAnnotation(
            folder=f'VOCdevkit/{folder}/JPEGImages',
            filename=filename,
            width=width,
            height=height,
            classes=classes,
            bboxes=bboxes,
            difficults=difficults)
        return annotation


def listup_classification(dirpath, class_names=None):
    """画像分類でよくある、クラス名ディレクトリの列挙。クラス名の配列, X, yを返す。"""
    dirpath = pathlib.Path(dirpath)

    # クラス名
    if class_names is None:
        def _empty(it):
            for _ in it:
                return False
            return True

        class_names = list(sorted([p.name for p in dirpath.iterdir() if p.is_dir() and not _empty(p.iterdir())]))

    # 各クラスのデータ
    X, y = [], []
    for class_id, class_name in enumerate(class_names):
        t = [p for p in (dirpath / class_name).iterdir() if p.is_file()]
        X.extend(t)
        y.extend([class_id] * len(t))
    assert len(X) == len(y)
    return class_names, np.array(X), np.array(y)


def split(X, y, validation_split=None, cv_count=None, cv_index=None, split_seed=None, stratify=None):
    """データの分割。

    # 引数
    - validation_split: 実数を指定するとX, y, weightsの一部をランダムにvalidation dataとする
    - cv_count: cross validationする場合の分割数
    - cv_index: cross validationする場合の何番目か
    - split_seed: validation_splitやcvする場合のseed
    """
    assert len(X) == len(y)
    if validation_split is not None:
        # split
        assert cv_count is None
        assert cv_index is None
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X, y, test_size=validation_split, shuffle=True, random_state=split_seed, stratify=stratify)
    else:
        # cross validation
        assert cv_count is not None
        assert cv_index in range(cv_count)
        if stratify is None:
            stratify = isinstance(y, np.ndarray) and len(y.shape) == 1
        cv = sklearn.model_selection.StratifiedKFold if stratify else sklearn.model_selection.KFold
        cv = cv(cv_count, shuffle=True, random_state=split_seed)
        train_indices, val_indices = list(cv.split(X, y))[cv_index]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
    return (X_train, y_train), (X_val, y_val)


class _ToCategorical(object):
    """クラスラベルのone-hot encoding化を行うクラス。"""

    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    def __call__(self, y):
        assert len(y.shape) == 1
        cat = np.zeros((len(y), self.nb_classes), dtype=np.float32)
        cat[np.arange(len(y)), y] = 1
        return cat


def to_categorical(nb_classes):
    """クラスラベルのone-hot encoding化を行う関数を返す。"""
    return _ToCategorical(nb_classes)


def compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                pred_classes_list, pred_confs_list, pred_bboxes_list,
                iou_threshold=0.5, use_voc2007_metric=False):
    """`mAP`の算出。

    # 引数
    - gt_classes_list: 正解のbounding box毎のクラス。shape=(画像数, bbox数)
    - gt_bboxes_list: 正解のbounding boxの座標(x1,y1,x2,y2。0～1)。shape=(画像数, bbox数, 4)
    - gt_difficults_list: 正解のbounding boxにdifficultフラグがついているか否か。shape=(画像数, bbox数)
    - pred_classes_list: 予測結果のbounding box毎のクラス。shape=(画像数, bbox数)
    - pred_confs_list: 予測結果のbounding box毎のconfidence。shape=(画像数, bbox数)
    - pred_bboxes_list: 予測結果のbounding boxの座標(x1,y1,x2,y2。0～1)。shape=(画像数, bbox数, 4)

    """
    assert len(gt_classes_list) == len(gt_bboxes_list)
    assert len(gt_classes_list) == len(gt_difficults_list)
    assert len(gt_classes_list) == len(pred_classes_list)
    assert len(gt_classes_list) == len(pred_confs_list)
    assert len(gt_classes_list) == len(pred_bboxes_list)

    npos_dict = collections.defaultdict(int)
    scores = collections.defaultdict(list)
    matches = collections.defaultdict(list)

    for gt_classes, gt_bboxes, gt_difficults, pred_classes, pred_confs, pred_bboxes in zip(
            gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_confs_list, pred_bboxes_list):

        for class_id in np.unique(np.concatenate([gt_classes, pred_classes])):
            pred_mask = pred_classes == class_id
            pred_confs_class = pred_confs[pred_mask]
            pred_bboxes_class = pred_bboxes[pred_mask]
            order = pred_confs_class.argsort()[::-1]
            pred_confs_class = pred_confs_class[order]
            pred_bboxes_class = pred_bboxes_class[order]

            gt_mask = gt_classes == class_id
            gt_bboxes_class = gt_bboxes[gt_mask]
            gt_difficults_class = gt_difficults[gt_mask]

            npos_dict[class_id] += np.logical_not(gt_difficults_class).sum()
            scores[class_id].extend(pred_confs_class)

            if len(pred_bboxes_class) == 0:
                continue
            if len(gt_bboxes_class) == 0:
                matches[class_id].extend([0] * len(pred_bboxes_class))
                continue

            iou = compute_iou(pred_bboxes_class, gt_bboxes_class)
            gt_indices = iou.argmax(axis=1)
            gt_indices[iou.max(axis=1) < iou_threshold] = -1  # 不一致

            detected = np.zeros(len(gt_bboxes_class), dtype=bool)
            for gt_i in gt_indices:
                if gt_i >= 0:
                    if gt_difficults_class[gt_i]:
                        matches[class_id].append(-1)  # skip
                    else:
                        matches[class_id].append(0 if detected[gt_i] else 1)
                        detected[gt_i] = True  # 2回目以降は不一致扱い
                else:
                    matches[class_id].append(0)  # 不一致

    precision = []
    recall = []
    for class_id, npos in npos_dict.items():
        order = np.array(scores[class_id]).argsort()[::-1]
        matches_class = np.array(matches[class_id])[order]

        tp = np.cumsum(matches_class == 1)
        fp = np.cumsum(matches_class == 0)
        tpfp = fp + tp
        tpfp = np.where(tpfp == 0, [np.nan] * len(tpfp), tpfp)  # 警告対策 (0ならnanにする)

        precision.append(tp / tpfp)
        recall.append(None if npos == 0 else tp / npos)

    ap = [compute_ap(prec, rec, use_voc2007_metric) for prec, rec in zip(precision, recall)]
    return np.nanmean(ap)


def compute_ap(precision, recall, use_voc2007_metric=False):
    """Average precisionの算出。"""
    if precision is None or recall is None:
        return np.nan
    if use_voc2007_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(precision)[recall >= t])
            ap += p
        ap /= 11
    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # pylint: disable=no-member

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def top_k_accuracy(y_true, proba_pred, k=5):
    """Top-K accuracy。"""
    assert len(y_true.shape) == 1
    assert len(proba_pred.shape) == 2
    best_k = np.argsort(proba_pred, axis=1)[:, -k:]
    return np.mean([y in best_k[i, :] for i, y in enumerate(y_true)])


def bboxes_center(bboxes):
    """Bounding boxの中心を返す。"""
    assert bboxes.shape[-1] == 4
    return (bboxes[..., 2:] + bboxes[..., :2]) / 2


def bboxes_size(bboxes):
    """Bounding boxのサイズを返す。"""
    assert bboxes.shape[-1] == 4
    return bboxes[..., 2:] - bboxes[..., :2]


def bboxes_area(bboxes):
    """Bounding boxの面積を返す。"""
    return bboxes_size(bboxes).prod(axis=-1)


def compute_iou(bboxes_a, bboxes_b):
    """IoU(Intersection over union、Jaccard係数)の算出。

    重なり具合を示す係数。(0～1)
    """
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


def compute_size_based_iou(bbox_sizes_a, bbox_sizes_b):
    """中心が一致している場合のIoUを算出する。

    例えば(10, 15)と(15, 10)で0.5。
    """
    assert bbox_sizes_a.shape[0] > 0
    assert bbox_sizes_b.shape[0] > 0
    if bbox_sizes_a.shape[-1] == 4 and bbox_sizes_b.shape[-1] == 4:
        bbox_sizes_a = bbox_sizes_a[:, 2:] - bbox_sizes_a[:, :2]
        bbox_sizes_b = bbox_sizes_b[:, 2:] - bbox_sizes_b[:, :2]
    assert bbox_sizes_a.shape == (len(bbox_sizes_a), 2)
    assert bbox_sizes_b.shape == (len(bbox_sizes_b), 2)
    area_inter = np.prod(np.minimum(bbox_sizes_a[:, np.newaxis, :], bbox_sizes_b), axis=-1)
    area_a = np.prod(bbox_sizes_a, axis=-1)
    area_b = np.prod(bbox_sizes_b, axis=-1)
    area_union = area_a[:, np.newaxis] + area_b - area_inter
    iou = area_inter / area_union
    assert iou.shape == (len(bbox_sizes_a), len(bbox_sizes_b))
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


def cluster_by_iou(X, n_clusters, **kwargs):
    """`1 - IoU`の値によるクラスタリング。

    YOLOv2のDimension Clustersで使用されているもの。
    KMeansのモデルのインスタンスを返す。
    """
    def _iou_distances(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
        """`1 - IoU`を返す。"""
        if Y is None:
            return _iou_distances(X, X, Y_norm_squared, squared, X_norm_squared)
        if len(X.shape) == 1:
            return _iou_distances(np.expand_dims(X, 0), Y, Y_norm_squared, squared, X_norm_squared)
        assert X.shape == (len(X), 2)
        assert Y.shape == (len(Y), 2)
        dist = 1 - compute_size_based_iou(X, Y)
        return np.square(dist) if squared else dist

    from sklearn.cluster import k_means_
    old = k_means_.euclidean_distances
    k_means_.euclidean_distances = _iou_distances  # monkey-patch
    model = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs)
    model.fit(X)
    k_means_.euclidean_distances = old
    return model


def non_maximum_suppression(boxes, scores, top_k=200, iou_threshold=0.45):
    """`iou_threshold`分以上重なっている場合、スコアの大きい方のみ採用する。

    # 引数
    - boxes: ボックスの座標。shape=(ボックス数, 4)
    - scores: スコア。shape=(ボックス数,)
    - top_k: 最大何個の結果を返すか。
    - iou_threshold: 重なりの閾値。

    # 戻り値
    インデックス。

    """
    assert len(boxes) == len(scores)
    assert boxes.shape == (len(scores), 4)
    sorted_indices = scores.argsort()[::-1]
    boxes = boxes[sorted_indices]
    boxes_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

    selected = []
    for i, b in enumerate(boxes):
        if (b[2:] - b[:2] <= 0).any():
            continue  # 面積のないboxはskip
        if len(selected) == 0:
            selected.append(i)
        else:
            lt = np.maximum(b[:2], boxes[selected, :2])
            rb = np.minimum(b[2:], boxes[selected, 2:])
            area_inter = np.prod(rb - lt, axis=1) * (lt < rb).all(axis=1)
            iou = area_inter / (boxes_area[i] + boxes_area[selected] - area_inter)
            if (iou >= iou_threshold).any():
                continue  # 重なっているのでskip
            selected.append(i)
            if len(selected) >= top_k:
                break

    return sorted_indices[selected]


class WeakModel(object):
    """CVしたりout-of-folds predictionを作ったりするクラス。"""

    def __init__(self, model_dir, base_estimator, cv=5, fit_params=None):
        self.model_dir = pathlib.Path(model_dir)
        self.base_estimator = base_estimator
        self.cv = cv
        self.fit_params = fit_params
        self.estimators_ = None
        self.data_ = {}

    def fit(self, X, y, groups=None, pool=None):
        """学習"""
        if not pool:
            pool = mp.Pool()
        func, args = self.make_fit_tasks(X, y, groups)
        pool.map(func, args)

    def make_fit_tasks(self, X, y, groups=None):
        """学習の処理を作って返す。(func, args)形式。"""
        self._init_data()
        args = []
        for fold in range(self.cv):
            estimator = sklearn.base.clone(self.base_estimator)
            args.append((estimator, fold, X, y, groups))
        return self._fit, args

    def split(self, fold, X, y, groups=None):
        """データの分割"""
        rs = np.random.RandomState(self.data_['split_seed'])
        classifier = sklearn.base.is_classifier(self.base_estimator)
        # cv = sklearn.model_selection.check_cv(self.cv, y, classifier=classifier)
        if classifier:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=rs)
        else:
            cv = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=rs)
        return list(cv.split(X, y, groups))[fold]

    def _fit(self, estimator, fold, X, y, groups=None):
        """学習。"""
        X, y, groups = sklearn.utils.indexable(X, y, groups)  # pylint: disable=E0632
        fit_params = self.fit_params if self.fit_params is not None else {}

        train, _ = self.split(fold, X, y, groups)
        estimator.fit(X[train], y[train], **fit_params)
        pred = estimator.predict_proba(X)
        joblib.dump(pred, str(self.model_dir / f'predict.fold{fold}.train.pkl'))
        joblib.dump(estimator, str(self.model_dir / f'model.fold{fold}.pkl'))

    def oopf(self, X, y, groups=None):
        """out-of-folds predictionを作って返す。

        Xはデータの順序が変わってないかのチェック用。
        """
        self._init_data()
        oopf = joblib.load(str(self.model_dir / f'predict.fold{0}.train.pkl'))
        for fold in range(1, self.cv):
            pred = joblib.load(str(self.model_dir / f'predict.fold{fold}.train.pkl'))
            _, test = self.split(fold, X, y, groups)
            oopf[test] = pred[test]
        return oopf

    def _init_data(self):
        """self.data_の初期化。今のところ(?)split_seedのみ。"""
        model_json_file = self.model_dir / 'model.json'
        if model_json_file.is_file():
            # あれば読み込み
            with model_json_file.open() as f:
                self.data_ = json.load(f)
        else:
            # 無ければ生成して保存
            self.data_ = {
                'split_seed': int(np.random.randint(0, 2 ** 31)),
            }
            with model_json_file.open('w') as f:
                json.dump(self.data_, f, indent=4, sort_keys=True)


def plot_cm(cm, to_file='confusion_matrix.png', classes=None, normalize=True, title='Confusion matrix'):
    """Confusion matrixを画像化する。

    参考: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if classes is None:
        classes = [f'class {i}' for i in range(len(cm))]

    if normalize:
        cm = np.array(cm, dtype=np.float32)
        cm /= cm.sum(axis=1)[:, np.newaxis]  # normalize
    else:
        cm = np.array(cm)

    size = (len(cm) + 1) // 2
    fig = draw.create_figure(figsize=(size + 4, size + 2), dpi=96)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    fig.colorbar(res)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)

    to_file = pathlib.Path(to_file)
    to_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(to_file), bbox_inches='tight')
    fig.clf()


def plot_objects(base_image, save_path, classes, confs, locs, class_names):
    """画像＋オブジェクト([class_id + confidence + xmin/ymin/xmax/ymax]×n)を画像化する。

    # 引数
    - base_image: 元画像ファイルのパスまたはndarray
    - save_path: 保存先画像ファイルのパス
    - classes: クラスIDのリスト
    - confs: confidenceのリスト (None可)
    - locs: xmin/ymin/xmax/ymaxのリスト (それぞれ0.0 ～ 1.0)
    - class_names: クラスID→クラス名のリスト  (None可)

    """
    import io
    import matplotlib
    import matplotlib.cm

    if confs is None:
        confs = [None] * len(classes)
    assert len(classes) == len(confs)
    assert len(classes) == len(locs)
    if class_names is not None and any(classes):
        assert 0 <= np.min(classes) < len(class_names)
        assert 0 <= np.max(classes) < len(class_names)

    colors = matplotlib.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()

    if isinstance(base_image, np.ndarray):
        img = base_image
    elif isinstance(base_image, (str, pathlib.Path, io.IOBase)):
        import PIL.Image
        with PIL.Image.open(base_image) as pil:
            if pil.mode != 'RGB':
                pil = pil.convert('RGB')
            img = np.asarray(pil).astype(np.float32)
    else:
        raise ValueError(f'type error: type(base_image)={type(base_image)}')

    fig = draw.create_figure(dpi=96)
    ax = fig.add_subplot(111)
    ax.imshow(img / 255.)
    for classid, conf, loc in zip(classes, confs, locs):
        xmin = int(round(loc[0] * img.shape[1]))
        ymin = int(round(loc[1] * img.shape[0]))
        xmax = int(round(loc[2] * img.shape[1]))
        ymax = int(round(loc[3] * img.shape[0]))
        label = class_names[classid] if class_names is not None else str(classid)
        if conf is None:
            txt = label
        else:
            txt = f'{conf:0.2f}, {label}'
        color = colors[classid]
        ax.add_patch(matplotlib.patches.Rectangle(
            (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1,
            fill=False, edgecolor=color, linewidth=2))
        ax.text(xmin, ymin, txt, bbox={'facecolor': color, 'alpha': 0.5})

    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    fig.clf()
