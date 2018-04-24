"""お手製Object detectionのためのPrior boxの集合を管理するクラス。"""

import numpy as np
import sklearn.cluster
import sklearn.metrics

from . import losses
from .. import log, math, ml, utils

_VAR_LOC = 0.2  # SSD風(?)適当スケーリング
_VAR_SIZE = 0.2  # SSD風適当スケーリング


class PriorBoxes(object):
    """Prior boxの集合を管理するクラス。 """

    def __init__(self, map_sizes):
        # 出力するfeature mapのサイズ(降順)。例：[40, 20, 10]なら、40x40、20x20、10x10の出力を持つ
        self.map_sizes = np.array(sorted(map_sizes)[::-1])
        # feature mapのグリッドサイズに対する、prior boxの基準サイズの割合(面積昇順)。[1.5, 0.5] なら横が1.5倍、縦が0.5倍。
        self.pb_size_patterns = np.empty((0,))
        # shape=(box数, 4)で座標(アスペクト比などを適用する前のグリッド)
        self.pb_grid = np.empty((0,))
        # shape=(box数, 4)で座標
        self.pb_locs = np.empty((0,))
        # shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_info_indices = np.empty((0,))
        # shape=(box数,2)でprior boxの中心の座標
        self.pb_centers = np.empty((0,))
        # shape=(box数,2)でprior boxのサイズ
        self.pb_sizes = np.empty((0,))
        # shape=(box数,)で、有効なprior boxなのか否か。
        self.pb_mask = np.empty((0,))
        # 各prior boxの情報をdictで保持
        self.pb_info = []

    def to_dict(self):
        """保存。"""
        return self.__dict__

    def from_dict(self, data: dict):
        """読み込み。"""
        try:
            self.map_sizes = np.array(data['map_sizes'])
            self.pb_size_patterns = np.array(data['pb_size_patterns'])
            self.pb_grid = np.array(data['pb_grid'])
            self.pb_locs = np.array(data['pb_locs'])
            self.pb_info_indices = np.array(data['pb_info_indices'])
            self.pb_centers = np.array(data['pb_centers'])
            self.pb_sizes = np.array(data['pb_sizes'])
            self.pb_mask = np.array(data['pb_mask'])
            self.pb_info = data['pb_info']
        except KeyError as e:
            raise ValueError(f'PriorBoxes load error: data={data}') from e
        self._check_state()

    @log.trace()
    def fit(self, input_size, y_train: [ml.ObjectsAnnotation], pb_size_pattern_count=8):
        """訓練データからパラメータを適当に決めてインスタンスを作成する。

        # 引数
        - y_train: 訓練データ
        - input_size: 入力画像の縦幅と横幅のタプル
        - pb_size_pattern_count: feature mapごとに何種類のサイズのprior boxを作るか。

        """
        logger = log.get(__name__)
        logger.info(f'objects per image:    {np.mean([len(y.bboxes) for y in y_train]):.1f}')
        logger.info(f'difficults per image: {np.mean([np.sum(y.difficults) for y in y_train]):.1f}')

        # bboxのサイズ
        bboxes = np.concatenate([y.bboxes for y in y_train])
        bboxes_sizes = bboxes[:, 2:] - bboxes[:, :2]
        assert (bboxes_sizes >= 0).all()  # 不正なデータは含まれない想定 (手抜き)
        assert (bboxes_sizes < 1).all()  # 不正なデータは含まれない想定 (手抜き)

        # bboxのサイズごとにどこかのfeature mapに割り当てたことにして相対サイズをリストアップ
        tile_sizes = 1 / self.map_sizes
        min_area_pattern = 0.5  # タイルの半分を下限としてみる
        max_area_pattern = 1 / tile_sizes.max()  # 一番荒いmapが画像全体になるくらいのスケールを上限としてみる
        bboxes_size_patterns = []
        for tile_size in tile_sizes:
            size_pattern = bboxes_sizes / tile_size
            area_pattern = np.sqrt(size_pattern.prod(axis=-1))  # 縦幅・横幅の相乗平均のリスト
            mask = math.between(area_pattern, min_area_pattern, max_area_pattern)
            bboxes_size_patterns.append(size_pattern[mask])
        bboxes_size_patterns = np.concatenate(bboxes_size_patterns)
        assert len(bboxes_size_patterns.shape) == 2
        assert bboxes_size_patterns.shape[1] == 2

        # リストアップしたものをクラスタリング。(YOLOv2のDimension Clustersのようなもの)。
        cluster = sklearn.cluster.KMeans(n_clusters=pb_size_pattern_count, n_jobs=-1, random_state=123)
        pb_size_patterns = cluster.fit(bboxes_size_patterns).cluster_centers_
        assert pb_size_patterns.shape == (pb_size_pattern_count, 2)
        self.pb_size_patterns = pb_size_patterns[pb_size_patterns.prod(axis=-1).argsort(), :].astype(np.float32)  # 面積昇順

        # prior boxのサイズなどを算出
        self.pb_grid = []
        self.pb_locs = []
        self.pb_info_indices = []
        self.pb_centers = []
        self.pb_sizes = []
        for map_size in self.map_sizes:
            # 敷き詰める間隔
            tile_size = 1.0 / map_size
            # 敷き詰めたときの中央の位置のリスト
            lin = np.linspace(0.5 * tile_size, 1 - 0.5 * tile_size, map_size, dtype=np.float32)
            # 縦横に敷き詰め
            centers_x, centers_y = np.meshgrid(lin, lin)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            prior_boxes_center = np.concatenate((centers_x, centers_y), axis=1)
            # (x, y) → タイル×(x1, y1, x2, y2)
            prior_boxes_center = np.tile(prior_boxes_center, (1, 2))
            assert prior_boxes_center.shape == (map_size ** 2, 4)
            # グリッド
            grid = np.copy(prior_boxes_center)
            grid[:, :2] -= tile_size / 2
            grid[:, 2:] += tile_size / 2 + (1 / np.array(input_size))
            # パターンごとにループ
            for pb_pat in pb_size_patterns:
                # prior boxのサイズの基準
                pb_size = tile_size * pb_pat
                # prior boxのサイズ
                # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
                prior_boxes = np.copy(prior_boxes_center)
                prior_boxes[:, :2] -= pb_size / 2
                prior_boxes[:, 2:] += pb_size / 2 + (1 / np.array(input_size))
                # 追加
                self.pb_grid.extend(grid)
                self.pb_locs.extend(prior_boxes)
                self.pb_centers.extend(prior_boxes_center[:, :2])
                self.pb_sizes.extend([pb_size] * len(prior_boxes))
                self.pb_info_indices.extend([len(self.pb_info)] * len(prior_boxes))
                self.pb_info.append({
                    'map_size': map_size,
                    'size': np.sqrt(pb_size.prod()),
                    'aspect_ratio': pb_size[0] / pb_size[1],
                    'count': len(prior_boxes),
                })
        self.pb_grid = np.array(self.pb_grid)
        self.pb_locs = np.array(self.pb_locs)
        self.pb_info_indices = np.array(self.pb_info_indices)
        self.pb_centers = np.array(self.pb_centers)
        self.pb_sizes = np.array(self.pb_sizes)
        # はみ出ているprior boxは使用しないようにする
        self.pb_mask = math.between(self.pb_locs, -0.1, +1.1).all(axis=-1)
        # 結果のチェック
        self._check_state()

    def _check_state(self):
        """selfの状態が正しいかチェック。"""
        # shapeの確認
        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4), f'shape error: {self.pb_locs.shape}'
        assert self.pb_info_indices.shape == (nb_pboxes,), f'shape error: {self.pb_info_indices.shape}'
        assert self.pb_centers.shape == (nb_pboxes, 2), f'shape error: {self.pb_centers.shape}'
        assert self.pb_sizes.shape == (nb_pboxes, 2), f'shape error: {self.pb_sizes.shape}'
        assert self.pb_mask.shape == (nb_pboxes,), f'shape error: {self.pb_mask.shape}'
        assert len(self.pb_info) == len(self.map_sizes) * len(self.pb_size_patterns)
        # prior boxの重心はpb_gridの中心であるはず
        ct = np.mean([self.pb_locs[:, 2:], self.pb_locs[:, :2]], axis=0)
        assert math.in_range(ct, self.pb_grid[:, :2], self.pb_grid[:, 2:]).all()

    def summary(self, logger=None):
        """サマリ表示。"""
        logger = logger or log.get(__name__)
        # feature mapのグリッドサイズに対する、prior boxの基準サイズの割合。(縦横の相乗平均)
        logger.info(f'prior box size ratios:   {np.sort(np.sqrt(self.pb_size_patterns[:, 0] * self.pb_size_patterns[:, 1]))}')
        # アスペクト比のリスト
        logger.info(f'prior box aspect ratios: {np.sort(self.pb_size_patterns[:, 0] / self.pb_size_patterns[:, 1])}')
        # サイズのリスト
        logger.info(f'prior box sizes: {np.unique([c["size"] for c in self.pb_info])}')
        # 数
        logger.info(f'prior box count: {len(self.pb_mask)} (valid={np.count_nonzero(self.pb_mask)})')

    def encode_locs(self, bboxes, bb_ix, pb_ix):
        """座標を学習用に変換。"""
        return (bboxes[bb_ix, :] - self.pb_locs[pb_ix, :]) / np.tile(self.pb_sizes[pb_ix, :], 2) / _VAR_LOC

    def decode_locs(self, pred, xp=None):
        """encode_locsの逆変換。xpはnumpy or keras.backend。"""
        if xp is None:
            import keras.backend as K
            xp = K
        decoded = pred * (_VAR_LOC * np.tile(self.pb_sizes, 2)) + self.pb_locs
        return xp.clip(decoded, 0, 1)

    def check_prior_boxes(self, y_test: [ml.ObjectsAnnotation], num_classes):
        """データに対してprior boxがどれくらいマッチしてるか調べる。"""
        y_true = []
        y_pred = []
        total_errors = 0
        iou_list = []
        assigned_counts = np.zeros((len(self.pb_info),))
        assigned_count_list = []
        delta_locs = []  # Δlocs

        total_gt_boxes = sum([np.sum(np.logical_not(y.difficults)) for y in y_test])

        for y in utils.tqdm(y_test, desc='check_prior_boxes'):
            # 割り当ててみる
            assigned_pb_list, assigned_gt_list, _ = self._assign_boxes(y.bboxes)
            # 1画像あたり何件のprior boxにassignされたか
            assigned_count_list.append(len(assigned_pb_list))
            # 初期の座標のずれ具合の集計
            for assigned_pb, assigned_gt in zip(assigned_pb_list, assigned_gt_list):
                delta_locs.append(self.encode_locs(y.bboxes, assigned_gt, assigned_pb))
            # オブジェクトごとの集計
            for gt_ix, (class_id, difficult) in enumerate(zip(y.classes, y.difficults)):
                assert 0 <= class_id < num_classes
                if difficult:
                    continue
                gt_mask = assigned_gt_list == gt_ix
                if gt_mask.any():
                    gt_pb = assigned_pb_list[gt_mask]
                    gt_iou = ml.compute_iou(y.bboxes[gt_ix:gt_ix + 1], self.pb_locs[gt_pb])[0]
                    max_iou = gt_iou.max()
                    pb_ix = gt_pb[gt_iou.argmax()]
                    # prior box毎の、割り当てられた回数
                    assigned_counts[self.pb_info_indices[pb_ix]] += 1
                else:
                    max_iou = 0
                    # 割り当て失敗
                    total_errors += 1
                # 再現率
                y_true.append(class_id)
                y_pred.append(class_id if max_iou >= 0.5 else None)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                # 最大IoU
                iou_list.append(max_iou)

        logger = log.get(__name__)
        # prior box毎の集計
        logger.info('assigned counts:')
        for i, c in enumerate(assigned_counts):
            logger.info('  prior boxes{m=%d, size=%.2f ar=%.2f} = %d (%.02f%%)',
                        self.pb_info[i]['map_size'],
                        self.pb_info[i]['size'],
                        self.pb_info[i]['aspect_ratio'],
                        c, 100 * c / self.pb_info[i]['count'] / total_gt_boxes)
        # 再現率
        logger.info('recall (prior box iou >= 0.5):')
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        for class_id in range(num_classes):
            y_mask = y_true == class_id
            class_ok, class_all = (y_pred[y_mask] == class_id).sum(), y_mask.sum()
            logger.info(f'  class{class_id:02d} : {class_ok:5d} / {class_all:5d} = {100 * class_ok / class_all:5.1f}%')
        # assignできなかった件数
        logger.info('total errors: %d / %d (%.02f%%)',
                    total_errors, total_gt_boxes, 100 * total_errors / total_gt_boxes)
        # YOLOv2の論文のTable 1相当の値のつもり (複数のprior boxに割り当てたときの扱いがちょっと違いそう)
        logger.info('Avg IOU: %.1f', np.mean(iou_list) * 100)
        # ヒストグラム: 1枚の画像で何個のprior boxがassignされたか
        math.print_histgram(assigned_count_list, name='assigned_count', print_fn=logger.info)
        logger.info('assigned count per image: median=%d min=%d max=%d',
                    np.median(assigned_count_list), min(assigned_count_list), max(assigned_count_list))
        # ヒストグラム: Δlocの絶対値の分布
        # mean≒0, std≒1とかくらいが学習しやすいはず。(SSDを真似た謎のスケーリングを行う)
        math.print_histgram([np.mean(np.abs(dl)) for dl in delta_locs], name='mean_abs_delta', print_fn=logger.info)
        delta_locs = np.concatenate(delta_locs)
        logger.info('delta loc: mean=%.2f std=%.2f min=%.2f max=%.2f',
                    delta_locs.mean(), delta_locs.std(), delta_locs.min(), delta_locs.max())

    def encode_truth(self, y_gt: [ml.ObjectsAnnotation], num_classes):
        """学習用の`y_true`の作成。"""
        # mask, objs, clfs, locs
        y_true = np.zeros((len(y_gt), len(self.pb_locs), 1 + 1 + num_classes + 4), dtype=np.float32)
        for i, y in enumerate(y_gt):
            assert math.in_range(y.classes, 0, num_classes).all()
            assigned_pb_list, assigned_gt_list, pb_valids = self._assign_boxes(y.bboxes)
            y_true[i, pb_valids, 0] = 1  # 正例 or 負例なら1。微妙なのは0。
            for pb_ix, gt_ix in zip(assigned_pb_list, assigned_gt_list):
                y_true[i, pb_ix, 1] = 1  # 0:bg 1:obj
                y_true[i, pb_ix, 2 + y.classes[gt_ix]] = 1
                y_true[i, pb_ix, -4:] = self.encode_locs(y.bboxes, gt_ix, pb_ix)
        assert (np.logical_and(y_true[:, :, 0], y_true[:, :, 1]) == y_true[:, :, 1]).all()  # objであるなら常に有効
        return y_true

    def _assign_boxes(self, bboxes):
        """各bounding boxをprior boxに割り当てる。

        戻り値は、prior boxのindexとbboxesのindexのタプルのリスト。
        """
        assert len(bboxes) >= 1
        bb_centers = ml.bboxes_center(bboxes)

        pb_assigned_gt = -np.ones((len(self.pb_locs),), dtype=int)  # -1埋め
        pb_assigned_iou = np.zeros((len(self.pb_locs),), dtype=float)
        pb_assignable = np.ones((len(self.pb_locs),), dtype=bool)
        pb_valids = np.ones((len(self.pb_locs),), dtype=bool)

        # 面積の降順に並べ替え (おまじない: 出来るだけ小さいのが埋もれないように)
        sorted_indices = ml.bboxes_area(bboxes).argsort()[::-1]
        # とりあえずSSD風にIoU >= 0.5に割り当て
        for gt_ix, bbox, bb_center in zip(sorted_indices, bboxes[sorted_indices], bb_centers[sorted_indices]):
            # bboxの重心が含まれるprior boxにのみ割り当てる
            pb_mask = math.in_range(bb_center, self.pb_locs[:, :2], self.pb_locs[:, 2:]).all(axis=-1)
            pb_mask = np.logical_and(pb_mask, self.pb_mask)
            assert pb_mask.any(), f'Encode error: {bb_center}'
            # IoUが0.5以上のものに割り当てる。1つも無ければ最大のものに。
            iou = ml.compute_iou(np.expand_dims(bbox, axis=0), self.pb_locs[pb_mask, :])[0]
            iou_mask = iou >= 0.5
            if iou_mask.any():
                for pb_ix, pb_iou in zip(np.where(pb_mask)[0][iou_mask], iou[iou_mask]):
                    # よりIoUが大きいものを優先して割り当て
                    if pb_assignable[pb_ix] and pb_assigned_iou[pb_ix] < pb_iou:
                        pb_assigned_gt[pb_ix] = gt_ix
                        pb_assigned_iou[pb_ix] = pb_iou
            else:
                iou_ix = iou.argmax()
                pb_ix = np.where(pb_mask)[0][iou_ix]
                pb_assigned_gt[pb_ix] = gt_ix
                pb_assigned_iou[pb_ix] = iou[iou_ix]
                pb_assignable[pb_ix] = False  # 上書き禁止！
            # IoUが0.3～0.5のprior boxは無視する
            iou_ignore_mask = np.logical_and(np.logical_not(iou_mask), iou >= 0.3)
            pb_ignore_mask = np.where(pb_mask)[0][iou_ignore_mask]
            pb_valids[pb_ignore_mask] = False

        # 中心が一致している前提で最も一致するところに強制割り当て
        pb_assignable = np.ones((len(self.pb_locs),), dtype=bool)
        for gt_ix, (bbox, bb_center) in enumerate(zip(bboxes, bb_centers)):
            # bboxの重心が含まれるグリッドのみ探す
            pb_mask = math.in_range(bb_center, self.pb_grid[:, :2], self.pb_grid[:, 2:]).all(axis=-1)
            pb_mask = np.logical_and(pb_mask, self.pb_mask)
            assert pb_mask.any(), f'Encode error: {bb_center}'
            pb_mask = np.logical_and(pb_mask, pb_assignable)  # 割り当て済みは除外
            if not pb_mask.any():
                continue  # 割り当て失敗
            # 形が最も合うもの1つにassignする
            sb_iou = ml.compute_size_based_iou(np.expand_dims(bbox, axis=0), self.pb_locs[pb_mask])[0]
            sb_iou_ix = sb_iou.argmax()
            pb_ix = np.where(pb_mask)[0][sb_iou_ix]
            pb_assigned_gt[pb_ix] = gt_ix
            pb_assignable[pb_ix] = False

        pb_indices = np.where(pb_assigned_gt >= 0)[0]
        assert len(pb_indices) >= 1  # 1個以上は必ず割り当てないと損失関数などが面倒になる

        pb_valids[pb_indices] = True  # 割り当て済みのところは有効

        return pb_indices, pb_assigned_gt[pb_indices], pb_valids

    def loss(self, y_true, y_pred):
        """損失関数。"""
        import keras.backend as K
        loss_obj = self.loss_obj(y_true, y_pred)
        loss_clf = self.loss_clf(y_true, y_pred)
        loss_loc = self.loss_loc(y_true, y_pred)
        loss = loss_obj + loss_clf + loss_loc
        assert len(K.int_shape(loss)) == 1  # (None,)
        return loss

    @property
    def metrics(self):
        """各種metricをまとめて返す。"""
        import keras.backend as K

        def acc_bg(y_true, y_pred):
            """背景の再現率。"""
            gt_mask = y_true[:, :, 0]
            gt_obj, pred_obj = y_true[:, :, 1], y_pred[:, :, 1]
            gt_bg = (1 - gt_obj) * gt_mask   # 背景
            acc = K.cast(K.equal(K.greater(gt_obj, 0.5), K.greater(pred_obj, 0.5)), K.floatx())
            return K.sum(acc * gt_bg, axis=-1) / K.sum(gt_bg, axis=-1)

        def acc_obj(y_true, y_pred):
            """物体の再現率。"""
            gt_obj, pred_obj = y_true[:, :, 1], y_pred[:, :, 1]
            acc = K.cast(K.equal(K.greater(gt_obj, 0.5), K.greater(pred_obj, 0.5)), K.floatx())
            return K.sum(acc * gt_obj, axis=-1) / K.sum(gt_obj, axis=-1)

        return [self.loss_obj, self.loss_clf, self.loss_loc, acc_bg, acc_obj]

    def loss_obj(self, y_true, y_pred):
        """ObjectnessのFocal loss。"""
        import keras.backend as K
        import tensorflow as tf
        gt_mask = y_true[:, :, 0]
        gt_obj, pred_obj = y_true[:, :, 1], y_pred[:, :, 1]
        gt_obj_count = K.sum(gt_obj, axis=-1)  # 各batch毎のobj数。
        with tf.control_dependencies([tf.assert_positive(gt_obj_count)]):  # obj_countが1以上であることの確認
            gt_obj_count = tf.identity(gt_obj_count)
        pb_mask = np.expand_dims(self.pb_mask, axis=0)
        loss = losses.binary_focal_loss(gt_obj, pred_obj)
        loss = K.sum(loss * gt_mask * pb_mask, axis=-1) / gt_obj_count  # normalized by the number of anchors assigned to a ground-truth box
        return loss

    @staticmethod
    def loss_clf(y_true, y_pred):
        """クラス分類のloss。多クラスだけどbinary_crossentropy。(cf. YOLOv3)"""
        import keras.backend as K
        gt_obj = y_true[:, :, 1]
        gt_classes, pred_classes = y_true[:, :, 2:-4], y_pred[:, :, 2:-4]
        loss = K.binary_crossentropy(gt_classes, pred_classes)
        loss = K.sum(loss, axis=-1)  # sum(classes)
        loss = K.sum(gt_obj * loss, axis=-1) / K.sum(gt_obj, axis=-1)  # mean (box)
        return loss

    @staticmethod
    def loss_loc(y_true, y_pred):
        """位置のloss。"""
        import keras.backend as K
        gt_obj = y_true[:, :, 1]
        gt_locs, pred_locs = y_true[:, :, -4:], y_pred[:, :, -4:]
        loss = losses.l1_smooth_loss(gt_locs, pred_locs)
        loss = K.sum(loss, axis=-1)  # loss(x1) + loss(y1) + loss(x2) + loss(y2)
        loss = K.sum(gt_obj * loss, axis=-1) / K.sum(gt_obj, axis=-1)  # mean (box)
        return loss
