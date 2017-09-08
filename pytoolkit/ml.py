"""機械学習(主にsklearn)関連。"""
import itertools
import json
import multiprocessing as mp
import pathlib

import numpy as np
import sklearn.base
import sklearn.model_selection
import sklearn.utils


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
    xy1 = np.maximum(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2])
    xy2 = np.minimum(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    area_ab = np.prod(xy2 - xy1, axis=2) * (xy1 < xy2).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    return area_ab / (area_a[:, np.newaxis] + area_b - area_ab)


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
