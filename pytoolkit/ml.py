"""機械学習(主にsklearn)関連。"""
import collections
import itertools
import json
import multiprocessing as mp
import pathlib
import xml.etree

import numpy as np
import scipy.misc
import sklearn.base
import sklearn.model_selection
import sklearn.utils

# VOC2007などのアノテーションデータを持つためのnamedtuple
# classes、bboxes、difficultsはそれぞれbounding boxの数分の配列。
ObjectsAnnotation = collections.namedtuple(
    'ObjectsAnnotation', 'folder,filename,width,height,classes,bboxes,difficults')

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


def compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list,
                pred_classes_list, pred_bboxes_list,
                iou_threshold=0.5, use_voc2007_metric=False):
    """`mAP`の算出。

    - gt_classes_list: 正解のbounding box毎のクラス。shape=(画像数, bbox数)
    - gt_bboxes_list: 正解のbounding boxの座標(x1,y1,x2,y2。0～1)。shape=(画像数, bbox数, 4)
    - gt_difficults_list: 正解のbounding boxにdifficultフラグがついているか否か。shape=(画像数, bbox数)
    - pred_classes_list: 予測結果のbounding box毎のクラス。shape=(画像数, bbox数)
    - pred_bboxes_list: 予測結果のbounding boxの座標(x1,y1,x2,y2。0～1)。shape=(画像数, bbox数, 4)
    """
    assert len(gt_classes_list) == len(gt_bboxes_list)
    assert len(gt_classes_list) == len(gt_difficults_list)
    assert len(gt_classes_list) == len(pred_classes_list)
    assert len(gt_classes_list) == len(pred_bboxes_list)

    matches = []
    for gt_classes, gt_bboxes, gt_difficults, pred_classes, pred_bboxes in zip(
            gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_classes_list, pred_bboxes_list):
        if len(pred_bboxes) == 0:
            continue
        if len(gt_bboxes) == 0:
            matches.extend([0] * pred_bboxes.shape[0])
            continue

        iou = compute_iou(pred_bboxes, gt_bboxes)
        pred_indices = iou.argmax(axis=0)
        gt_indices = iou.argmax(axis=1)
        gt_indices[iou.max(axis=1) < iou_threshold] = -1  # 不一致

        detected = np.zeros(len(gt_bboxes), dtype=bool)
        for gt_i in gt_indices:
            if gt_i >= 0 and pred_classes[pred_indices[gt_i]] == gt_classes[gt_i]:
                if gt_difficults[gt_i]:
                    matches.append(-1)  # skip
                else:
                    matches.append(0 if detected[gt_i] else 1)
                detected[gt_i] = True  # 2回目以降は不一致扱い
            else:
                matches.append(0)  # 不一致

    npos = sum([np.logical_not(gt_difficults).sum() for gt_difficults in gt_difficults_list])
    matches = np.array(matches)
    tp = np.cumsum(matches == 1)
    fp = np.cumsum(matches == 0)
    recall = tp / npos
    precision = tp / (fp + tp)
    ap = compute_ap(recall, precision, use_voc2007_metric)
    return ap


def compute_ap(recall, precision, use_voc2007_metric=False):
    """Average precisionの算出。"""
    if use_voc2007_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(bboxes_a, bboxes_b):
    """IOU(Intersection over union、Jaccard係数)の算出。

    重なり具合を示す係数。(0～1)
    """
    assert bboxes_a.shape[0] > 0
    assert bboxes_b.shape[0] > 0
    assert bboxes_a.shape == (len(bboxes_a), 4)
    assert bboxes_b.shape == (len(bboxes_b), 4)
    xy1 = np.maximum(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2])
    xy2 = np.minimum(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    area_inter = np.prod(xy2 - xy1, axis=2) * (xy1 < xy2).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    area_union = area_a[:, np.newaxis] + area_b - area_inter
    return area_inter / area_union


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
        sklearn.externals.joblib.dump(pred, str(self.model_dir.joinpath('predict.fold{}.train.pkl'.format(fold))))
        sklearn.externals.joblib.dump(estimator, str(self.model_dir.joinpath('model.fold{}.pkl'.format(fold))))

    def oopf(self, X, y, groups=None):
        """out-of-folds predictionを作って返す。

        Xはデータの順序が変わってないかのチェック用。
        """
        self._init_data()
        oopf = sklearn.externals.joblib.load(str(self.model_dir.joinpath('predict.fold{}.train.pkl'.format(0))))
        for fold in range(1, self.cv):
            pred = sklearn.externals.joblib.load(str(self.model_dir.joinpath('predict.fold{}.train.pkl'.format(fold))))
            _, test = self.split(fold, X, y, groups)
            oopf[test] = pred[test]
        return oopf

    def _init_data(self):
        """self.data_の初期化。今のところ(?)split_seedのみ。"""
        model_json_file = self.model_dir.joinpath('model.json')
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
    import matplotlib.pyplot as plt

    if classes is None:
        classes = ['class {}'.format(i) for i in range(len(cm))]

    if normalize:
        cm = np.array(cm, dtype=np.float32)
        cm /= cm.sum(axis=1)[:, np.newaxis]  # normalize
    else:
        cm = np.array(cm)

    size = (len(cm) + 1) // 2
    fig = plt.figure(figsize=(size + 4, size + 2), dpi=96)
    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    fig.colorbar(res)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)

    plt.savefig(str(to_file), bbox_inches='tight')
    plt.close()


def load_voc_annotations(annotations_dir, class_name_to_id):
    """VOC2007などのアノテーションデータの読み込み。

    結果は「画像ファイル名拡張子なし」とObjectsAnnotationのdict。
    """
    data = {}
    for f in pathlib.Path(annotations_dir).iterdir():
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
            class_id = class_name_to_id[object_tree.find('name').text]
            bndbox = object_tree.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / width
            ymin = float(bndbox.find('ymin').text) / height
            xmax = float(bndbox.find('xmax').text) / width
            ymax = float(bndbox.find('ymax').text) / height
            difficult = object_tree.find('difficult').text == '1'
            classes.append(class_id)
            bboxes.append([xmin, ymin, xmax, ymax])
            difficults.append(difficult)
        data[f.stem] = ObjectsAnnotation(
            folder=folder,
            filename=filename,
            width=width,
            height=height,
            classes=np.array(classes),
            bboxes=np.array(bboxes),
            difficults=np.array(difficults))
    return data


def plot_objects(base_image_path, save_path, classes, confs, locs, class_names):
    """画像＋オブジェクト([class_id + confidence + xmin/ymin/xmax/ymax]×n)を画像化する。

    # 引数
    - base_image_path: 元画像ファイルのパス
    - save_path: 保存先画像ファイルのパス
    - classes: クラスIDのリスト
    - confs: confidenceのリスト (None可)
    - locs: xmin/ymin/xmax/ymaxのリスト (それぞれ0.0 ～ 1.0)
    - class_names: クラスID→クラス名のリスト  (None可)

    """
    import matplotlib.pyplot as plt

    if confs is None:
        confs = [None] * len(classes)
    assert len(classes) == len(confs)
    assert len(classes) == len(locs)
    if class_names is not None:
        assert 0 <= np.min(classes) < len(class_names)
        assert 0 <= np.max(classes) < len(class_names)

    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()

    img = scipy.misc.imread(str(base_image_path), mode='RGB')
    plt.imshow(img / 255.)
    gca = plt.gca()
    for classid, conf, loc in zip(classes, confs, locs):
        xmin = int(round(loc[0] * img.shape[1]))
        ymin = int(round(loc[1] * img.shape[0]))
        xmax = int(round(loc[2] * img.shape[1]))
        ymax = int(round(loc[3] * img.shape[0]))
        label = class_names[classid] if class_names else str(classid)
        if conf is None:
            txt = label
        else:
            txt = '{:0.2f}, {}'.format(conf, label)
        color = colors[classid]
        gca.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1,
            fill=False, edgecolor=color, linewidth=2))
        gca.text(xmin, ymin, txt, bbox={'facecolor': color, 'alpha': 0.5})

    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
