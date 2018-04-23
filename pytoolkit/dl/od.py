"""お手製Object detection。

https://github.com/ak110/object_detector
"""
import concurrent.futures
import pathlib

import numpy as np
import sklearn.cluster
import sklearn.externals.joblib as joblib
import sklearn.metrics

from . import layers, losses, models
from .. import generator, image, log, math, ml, ndimage, utils

_VAR_LOC = 0.2  # SSD風(?)適当スケーリング
_VAR_SIZE = 0.2  # SSD風適当スケーリング


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    @classmethod
    @log.trace()
    def create(cls, base_network, input_size, map_sizes, nb_classes, y_train, pb_size_pattern_count=8):
        """訓練データからパラメータを適当に決めてインスタンスを作成する。"""
        logger = log.get(__name__)
        logger.info('ハイパーパラメータ算出: base-network=%s input-size=%s map-sizes=%s classes=%d',
                    base_network, input_size, map_sizes, nb_classes)

        map_sizes = np.array(sorted(map_sizes)[::-1])
        mean_objets = np.mean([len(y.bboxes) for y in y_train])

        # bboxのサイズ
        bboxes = np.concatenate([y.bboxes for y in y_train])
        bboxes_sizes = bboxes[:, 2:] - bboxes[:, :2]
        bb_base_sizes = bboxes_sizes.min(axis=-1)  # 短辺のサイズのリスト
        assert (bb_base_sizes >= 0).all()  # 不正なデータは含まれない想定 (手抜き)
        assert (bb_base_sizes < 1).all()  # 不正なデータは含まれない想定 (手抜き)

        # bboxのサイズごとにどこかのfeature mapに割り当てたことにして相対サイズをリストアップ
        tile_sizes = 1 / map_sizes
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
        cluster = sklearn.cluster.KMeans(n_clusters=pb_size_pattern_count, n_jobs=-1)
        pb_size_patterns = cluster.fit(bboxes_size_patterns).cluster_centers_
        assert pb_size_patterns.shape == (pb_size_pattern_count, 2)

        return cls(base_network, input_size, map_sizes, nb_classes, mean_objets, pb_size_patterns)

    def __init__(self, base_network, input_size, map_sizes, nb_classes, mean_objets, pb_size_patterns):
        self.base_network = base_network
        self.image_size = (input_size, input_size)
        self.map_sizes = np.array(map_sizes)
        self.nb_classes = nb_classes
        # 1枚の画像あたりの平均オブジェクト数
        self.mean_objets = mean_objets
        # 1 / fm_countに対する、prior boxの基準サイズの割合。[1.5, 0.5] なら横が1.5倍、縦が0.5倍のものを用意。
        # 面積昇順に並べ替えておく。
        self.pb_size_patterns = pb_size_patterns[pb_size_patterns.prod(axis=-1).argsort(), :].astype(np.float32)
        # 1 / fm_countに対する、prior boxの基準サイズの割合。(縦横の相乗平均)
        self.pb_size_ratios = np.sort(np.sqrt(pb_size_patterns[:, 0] * pb_size_patterns[:, 1]))
        # アスペクト比のリスト
        self.pb_aspect_ratios = np.sort(pb_size_patterns[:, 0] / pb_size_patterns[:, 1])

        # shape=(box数, 4)で座標(アスペクト比などを適用する前のグリッド)
        self.pb_grid = []
        # shape=(box数, 4)で座標
        self.pb_locs = []
        # shape=(box数,)で、何種類目のprior boxか (集計用)
        self.pb_info_indices = []
        # shape=(box数,2)でprior boxの中心の座標
        self.pb_centers = []
        # shape=(box数,2)でprior boxのサイズ
        self.pb_sizes = []
        # 各prior boxの情報をdictで保持
        self.pb_info = []

        self._add_prior_boxes(self.map_sizes, self.pb_size_patterns)
        self.pb_grid = np.array(self.pb_grid)
        self.pb_locs = np.array(self.pb_locs)
        self.pb_info_indices = np.array(self.pb_info_indices)
        self.pb_centers = np.array(self.pb_centers)
        self.pb_sizes = np.array(self.pb_sizes)

        # prior boxの重心はpb_gridの中心であるはず
        ct = np.mean([self.pb_locs[:, 2:], self.pb_locs[:, :2]], axis=0)
        assert math.in_range(ct, self.pb_grid[:, :2], self.pb_grid[:, 2:]).all()
        # はみ出ているprior boxは使用しないようにする
        self.pb_mask = math.between(self.pb_locs, -0.1, +1.1).all(axis=-1)

        nb_pboxes = len(self.pb_locs)
        assert self.pb_locs.shape == (nb_pboxes, 4), f'shape error: {self.pb_locs.shape}'
        assert self.pb_info_indices.shape == (nb_pboxes,), f'shape error: {self.pb_info_indices.shape}'
        assert self.pb_centers.shape == (nb_pboxes, 2), f'shape error: {self.pb_centers.shape}'
        assert self.pb_sizes.shape == (nb_pboxes, 2), f'shape error: {self.pb_sizes.shape}'
        assert self.pb_mask.shape == (nb_pboxes,), f'shape error: {self.pb_mask.shape}'
        assert len(self.pb_info) == len(self.map_sizes) * len(self.pb_size_patterns)

    def _add_prior_boxes(self, map_sizes, pb_size_patterns):
        """基準となるbox(prior box)の座標(x1, y1, x2, y2)の並びを返す。"""
        for map_size in map_sizes:
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
            grid[:, 2:] += tile_size / 2 + (1 / np.array(self.image_size))

            for pb_pat in pb_size_patterns:
                # prior boxのサイズの基準
                pb_size = tile_size * pb_pat
                # prior boxのサイズ
                # (x1, y1, x2, y2)の位置を調整。縦横半分ずつ動かす。
                prior_boxes = np.copy(prior_boxes_center)
                prior_boxes[:, :2] -= pb_size / 2
                prior_boxes[:, 2:] += pb_size / 2 + (1 / np.array(self.image_size))

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

    def save(self, path: pathlib.Path):
        """保存。"""
        joblib.dump(self, path)

    @staticmethod
    def load(path: pathlib.Path):
        """読み込み。"""
        od = joblib.load(path)  # type: ObjectDetector
        od.summary()
        return od

    def summary(self, logger=None):
        """サマリ表示。"""
        logger = logger or log.get(__name__)
        logger.info('mean objects / image = %f', self.mean_objets)
        logger.info('prior box size ratios = %s', str(self.pb_size_ratios))
        logger.info('prior box aspect ratios = %s', str(self.pb_aspect_ratios))
        logger.info('prior box sizes = %s', str(np.unique([c['size'] for c in self.pb_info])))
        logger.info('prior box count = %d (valid=%d)', len(self.pb_mask), np.count_nonzero(self.pb_mask))

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

    def check_prior_boxes(self, y_test: [ml.ObjectsAnnotation], class_names):
        """データに対してprior boxがどれくらいマッチしてるか調べる。"""
        y_true = []
        y_pred = []
        total_errors = 0
        iou_list = []
        assigned_counts = np.zeros((len(self.pb_info),))
        assigned_count_list = []
        unrec_widths = []  # iou < 0.5の横幅
        unrec_heights = []  # iou < 0.5の高さ
        unrec_ars = []  # iou < 0.5のアスペクト比
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
                assert 0 <= class_id < self.nb_classes
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
                # 最大IoU
                iou_list.append(max_iou)
                # 再現率
                y_true.append(class_id)
                y_pred.append(class_id if max_iou >= 0.5 else 0)  # IOUが0.5以上のboxが存在すれば一致扱いとする
                # iou < 0.5しか無いboxの情報を集める
                if max_iou < 0.5:
                    bbox = y.bboxes[gt_ix]
                    wh = bbox[2:] - bbox[:2]
                    unrec_widths.append(wh[1])
                    unrec_heights.append(wh[0])
                    unrec_ars.append(wh[0] / wh[1])
        cr = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)

        # ログ出力
        logger = log.get(__name__)
        # ヒストグラム色々
        rec_mean_abs_delta = [np.mean(np.abs(dl)) for dl in delta_locs]
        math.print_histgram(unrec_widths, name='unrec_widths', print_fn=logger.info)
        math.print_histgram(unrec_heights, name='unrec_heights', print_fn=logger.info)
        math.print_histgram(unrec_ars, name='unrec_ars', print_fn=logger.info)
        math.print_histgram(rec_mean_abs_delta, name='rec_mean_abs_delta', print_fn=logger.info)
        math.print_histgram(assigned_count_list, name='assigned_count', print_fn=logger.info)
        # classification_report
        logger.info(cr)
        # prior box毎の集計など
        logger.info('assigned counts:')
        for i, c in enumerate(assigned_counts):
            logger.info('  prior boxes{m=%d, size=%.2f ar=%.2f} = %d (%.02f%%)',
                        self.pb_info[i]['map_size'],
                        self.pb_info[i]['size'],
                        self.pb_info[i]['aspect_ratio'],
                        c, 100 * c / self.pb_info[i]['count'] / total_gt_boxes)
        logger.info('total errors: %d / %d (%.02f%%)',
                    total_errors, total_gt_boxes, 100 * total_errors / total_gt_boxes)
        # iou < 0.5の出現率
        logger.info('[iou < 0.5] count: %d / %d (%.02f%%)',
                    len(unrec_widths), len(y_test), 100 * len(unrec_widths) / len(y_test))
        # YOLOv2の論文のTable 1相当の値のつもり (複数のprior boxに割り当てたときの扱いがちょっと違いそう)
        logger.info('Avg IOU: %.1f', np.mean(iou_list) * 100)
        # 1画像あたり何件のprior boxにassignされたか
        logger.info('assigned count per image: mean=%.1f std=%.2f min=%d max=%d',
                    np.mean(assigned_count_list), np.std(assigned_count_list),
                    min(assigned_count_list), max(assigned_count_list))
        # Δlocの分布調査
        # mean≒0, std≒1とかくらいが学習しやすいはず。(SSDを真似た謎の0.1で大体そうなってる)
        delta_locs = np.concatenate(delta_locs)
        logger.info('delta loc: mean=%.2f std=%.2f min=%.2f max=%.2f',
                    delta_locs.mean(), delta_locs.std(), delta_locs.min(), delta_locs.max())

    def encode_truth(self, y_gt: [ml.ObjectsAnnotation]):
        """学習用の`y_true`の作成。"""
        # mask, objs, clfs, locs
        y_true = np.zeros((len(y_gt), len(self.pb_locs), 1 + 1 + self.nb_classes + 4), dtype=np.float32)
        for i, y in enumerate(y_gt):
            assert math.in_range(y.classes, 0, self.nb_classes).all()
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

    def get_preprocess_input(self):
        """`preprocess_input`を返す。"""
        if self.base_network in ('vgg16', 'resnet50'):
            return image.preprocess_input_mean
        else:
            assert self.base_network in ('custom', 'xception')
            return image.preprocess_input_abs1

    @log.trace()
    def create_model(self, mode, batch_size, weights='voc', multi_gpu_predict=True):
        """学習とか予測とか用に`tk.dl.models.Model`を作成して返す。

        # 引数
        - mode: 'pretrain', 'train', 'predict'のいずれか。(出力などが違う)
        - batch_size: バッチサイズ。(`tk.dl.models.Model`に渡される)
        - weights: 読み込む重み。Noneなら読み込まない。'imagenet'ならバックボーンだけ。'voc'ならVOC07+12で学習したものを読み込む。pathlib.Pathならそのまま読み込む。

        """
        assert mode in ('pretrain', 'train', 'predict')
        assert weights in (None, 'imagenet', 'voc') or isinstance(weights, pathlib.Path)
        import keras
        builder = layers.Builder()
        x = inputs = keras.layers.Input(self.image_size + (3,))
        x, ref, lr_multipliers = self._create_basenet(builder, x, weights == 'imagenet')
        if mode == 'pretrain':
            assert len(lr_multipliers) == 0
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = builder.dense(4, activation='softmax', name='predictions')(x)
            model = keras.models.Model(inputs=inputs, outputs=x)
        else:
            model = self._create_detector(builder, inputs, x, ref, lr_multipliers, for_predict=mode == 'predict')

        logger = log.get(__name__)
        logger.info('network depth: %d', models.count_network_depth(model))
        logger.info('trainable params: %d', models.count_trainable_params(model))

        gen = self.create_pretrain_generator() if mode == 'pretrain' else self.create_generator()
        model = models.Model(model, gen, batch_size)

        if weights == 'voc':
            pass  # TODO: githubに学習済みモデル置いてkeras.applicationsみたいなダウンロード機能作る。

        if isinstance(weights, pathlib.Path):
            model.load_weights(weights)
            logger.info(f'warm start: {weights.name}')

        if mode == 'pretrain':
            # 事前学習：通常の分類としてコンパイル
            model.compile(sgd_lr=0.5 / 256, loss='categorical_crossentropy', metrics=['acc'])
        elif mode == 'train':
            # Object detectionとしてコンパイル
            sgd_lr = 0.5 / 256 / 3  # lossが複雑なので微調整
            model.compile(sgd_lr=sgd_lr, lr_multipliers=lr_multipliers, loss=self.loss, metrics=self.metrics)
        else:
            assert mode == 'predict'
            # 予測：コンパイル不要。マルチGPU化。
            if multi_gpu_predict:
                model.set_multi_gpu_model()
        return model

    @log.trace()
    def _create_basenet(self, builder, x, load_weights):
        """ベースネットワークの作成。"""
        import keras
        basenet = None
        ref_list = []
        if self.base_network == 'custom':
            x = builder.conv2d(32, 7, strides=2, name='stage0_ds')(x)
            x = keras.layers.MaxPooling2D(name='stage1_ds')(x)
            x = builder.conv2d(64, 3, name='stage2_conv1')(x)
            x = builder.conv2d(64, 3, name='stage2_conv2')(x)
            x = keras.layers.MaxPooling2D(name='stage2_ds')(x)
            x = builder.conv2d(128, 3, name='stage3_conv1')(x)
            x = builder.conv2d(128, 3, name='stage3_conv2')(x)
            x = keras.layers.MaxPooling2D(name='stage3_ds')(x)
            x = builder.conv2d(256, 3, name='stage4_conv1')(x)
            x = builder.conv2d(256, 3, name='stage4_conv2')(x)
            ref_list.append(x)
        elif self.base_network == 'vgg16':
            basenet = keras.applications.VGG16(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
            ref_list.append(basenet.get_layer(name='block4_pool').input)
            ref_list.append(basenet.get_layer(name='block5_pool').input)
        elif self.base_network == 'resnet50':
            basenet = keras.applications.ResNet50(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
            ref_list.append(basenet.get_layer(name='res4a_branch2a').input)
            ref_list.append(basenet.get_layer(name='res5a_branch2a').input)
            ref_list.append(basenet.get_layer(name='avg_pool').input)
        elif self.base_network == 'xception':
            basenet = keras.applications.Xception(include_top=False, input_tensor=x, weights='imagenet' if load_weights else None)
            ref_list.append(basenet.get_layer(name='block4_sepconv1_act').input)
            ref_list.append(basenet.get_layer(name='block13_sepconv1_act').input)
            ref_list.append(basenet.get_layer(name='block14_sepconv2_act').output)
        else:
            assert False

        # 転移学習元部分の学習率は控えめにする
        lr_multipliers = {}
        if basenet is not None:
            for layer in basenet.layers:
                w = layer.trainable_weights
                lr_multipliers.update(zip(w, [0.01] * len(w)))

        # チャンネル数が多ければここで減らす
        x = ref_list[-1]
        if builder.shape(x)[-1] > 256:
            x = builder.conv2d(256, (1, 1), name='tail_sq')(x)
        assert builder.shape(x)[-1] == 256

        # downsampling
        down_index = 0
        while True:
            down_index += 1
            map_size = builder.shape(x)[1] // 2
            x = builder.conv2d(256, 2, strides=2, name=f'down{down_index}_ds')(x)
            x = builder.conv2d(256, 3, name=f'down{down_index}_conv')(x)
            assert builder.shape(x)[1] == map_size
            ref_list.append(x)
            if map_size <= 4 or map_size % 2 != 0:  # 充分小さくなるか奇数になったら終了
                break

        ref = {f'down{builder.shape(x)[1]}': x for x in ref_list}
        return x, ref, lr_multipliers

    @log.trace()
    def _create_detector(self, builder, inputs, x, ref, lr_multipliers, for_predict):
        """ネットワークのcenter以降の部分を作る。"""
        import keras
        import keras.backend as K
        map_size = builder.shape(x)[1]

        # center
        x = builder.conv2d(32, (1, 1), name='center_conv1')(x)
        x = builder.conv2d(32, (map_size, map_size), padding='valid', name='center_conv2')(x)
        x = builder.conv2d(32, (1, 1), name='center_conv3')(x)

        # upsampling
        up_index = 0
        while True:
            up_index += 1
            in_map_size = builder.shape(x)[1]
            assert map_size % in_map_size == 0, f'map size error: {in_map_size} -> {map_size}'
            up_size = map_size // in_map_size
            x = keras.layers.Dropout(0.25)(x)
            x = builder.conv2dtr(64, up_size, strides=up_size, padding='valid', name=f'up{up_index}_us')(x)
            x = builder.conv2d(256, 1, use_act=False, name=f'up{up_index}_ex')(x)
            t = builder.conv2d(256, 1, use_act=False, name=f'up{up_index}_lt')(ref[f'down{map_size}'])
            x = keras.layers.add([x, t], name=f'up{up_index}_mix')
            x = builder.bn_act(name=f'up{up_index}_mix')(x)
            x = builder.conv2d(256, 3, name=f'up{up_index}_conv')(x)
            ref[f'out{map_size}'] = x

            if self.map_sizes[0] <= map_size:
                break
            map_size *= 2

        # prediction module
        objs, clfs, locs = self._create_pm(builder, ref, lr_multipliers)

        if for_predict:
            model = self._create_predict_network(inputs, objs, clfs, locs)
        else:
            # ラベル側とshapeを合わせるためのダミー
            dummy_shape = (len(self.pb_locs), 1)
            dummy = keras.layers.Lambda(K.zeros_like, dummy_shape, name='dummy')(objs)  # objsとちょうどshapeが同じなのでzeros_like。
            # いったんくっつける (損失関数の中で分割して使う)
            outputs = keras.layers.concatenate([dummy, objs, clfs, locs], axis=-1, name='outputs')
            model = keras.models.Model(inputs=inputs, outputs=outputs)

        return model

    @log.trace()
    def _create_pm(self, builder, ref, lr_multipliers):
        """Prediction module."""
        import keras

        old_gn, builder.use_gn = builder.use_gn, True

        shared_layers = {}
        shared_layers['pm_conv1'] = builder.conv2d(256, 3, use_act=False, name='pm_conv1')
        shared_layers['pm_conv2_1'] = builder.conv2d(256, 3, use_act=True, name='pm_conv2_1')
        shared_layers['pm_conv2_2'] = builder.conv2d(256, 3, use_act=False, name='pm_conv2_2')
        shared_layers['pm_conv3_1'] = builder.conv2d(256, 3, use_act=True, name='pm_conv3_1')
        shared_layers['pm_conv3_2'] = builder.conv2d(256, 3, use_act=False, name='pm_conv3_2')
        shared_layers['pm_bn_act'] = builder.bn_act(name='pm')
        for pat_ix in range(len(self.pb_size_patterns)):
            shared_layers[f'pm-{pat_ix}_obj'] = builder.conv2d(
                1, 1,
                bias_initializer=losses.od_bias_initializer(1),
                bias_regularizer=None,
                activation='sigmoid',
                use_bias=True,
                use_bn=False,
                name=f'pm-{pat_ix}_obj')
            shared_layers[f'pm-{pat_ix}_clf'] = builder.conv2d(
                self.nb_classes, 1,
                activation='sigmoid',  # softmaxより速そう (cf. YOLOv3)
                use_bias=True,
                use_bn=False,
                name=f'pm-{pat_ix}_clf')
            shared_layers[f'pm-{pat_ix}_loc'] = builder.conv2d(
                4, 1,
                use_bias=True,
                use_bn=False,
                use_act=False,
                name=f'pm-{pat_ix}_loc')
        for layer in shared_layers.values():
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [1 / len(self.map_sizes)] * len(w)))  # 共有部分の学習率調整

        builder.use_gn = old_gn

        objs, clfs, locs = [], [], []
        for map_size in self.map_sizes:
            assert f'out{map_size}' in ref, f'map_size error: {ref}'
            x = ref[f'out{map_size}']
            x = shared_layers[f'pm_conv1'](x)
            t = x
            x = shared_layers[f'pm_conv2_1'](x)
            x = shared_layers[f'pm_conv2_2'](x)
            x = keras.layers.add([t, x], name=f'pm{map_size}_mix1')
            t = x
            x = shared_layers[f'pm_conv2_1'](x)
            x = shared_layers[f'pm_conv2_2'](x)
            x = keras.layers.add([t, x], name=f'pm{map_size}_mix2')
            x = shared_layers[f'pm_bn_act'](x)
            for pat_ix in range(len(self.pb_size_patterns)):
                obj = shared_layers[f'pm-{pat_ix}_obj'](x)
                clf = shared_layers[f'pm-{pat_ix}_clf'](x)
                loc = shared_layers[f'pm-{pat_ix}_loc'](x)
                obj = keras.layers.Reshape((-1, 1), name=f'pm{map_size}-{pat_ix}_r1')(obj)
                clf = keras.layers.Reshape((-1, self.nb_classes), name=f'pm{map_size}-{pat_ix}_r2')(clf)
                loc = keras.layers.Reshape((-1, 4), name=f'pm{map_size}-{pat_ix}_r3')(loc)
                objs.append(obj)
                clfs.append(clf)
                locs.append(loc)
        objs = keras.layers.concatenate(objs, axis=-2, name='output_objs')
        clfs = keras.layers.concatenate(clfs, axis=-2, name='output_clfs')
        locs = keras.layers.concatenate(locs, axis=-2, name='output_locs')
        return objs, clfs, locs

    def _create_predict_network(self, inputs, objs, clfs, locs):
        """予測用ネットワークの作成"""
        import keras
        import keras.backend as K

        def _conf(x):
            objs = x[0][:, :, 0]
            confs = x[1]
            # objectnessとconfidenceの調和平均をconfidenceということにしてみる
            # conf = 2 / (1 / objs + 1 / confs)
            # → どうも相乗平均の方がmAP高いっぽい？
            conf = K.sqrt(objs * confs)
            return conf * np.expand_dims(self.pb_mask, axis=0)

        classes = layers.channel_argmax()(name='classes')(clfs)
        confs = layers.channel_max()(name='confs')(clfs)
        objconfs = keras.layers.Lambda(_conf, K.int_shape(clfs)[1:-1], name='objconfs')([objs, confs])
        locs = keras.layers.Lambda(self.decode_locs, name='locs')(locs)
        nms = layers.nms()(self.nb_classes, len(self.pb_locs), name='nms')([classes, objconfs, locs])
        return keras.models.Model(inputs=inputs, outputs=nms)

    def create_pretrain_generator(self):
        """ImageDataGeneratorを作って返す。"""
        gen = image.ImageDataGenerator()
        gen.add(image.Resize(self.image_size))
        gen.add(image.RandomFlipLR(probability=0.5))
        gen.add(image.RandomErasing(probability=0.5))
        gen.add(image.RotationsLearning())
        gen.add(image.ProcessInput(self.get_preprocess_input(), batch_axis=True))
        return gen

    def create_generator(self, encode_truth=True):
        """ImageDataGeneratorを作って返す。"""
        gen = image.ImageDataGenerator()
        gen.add(image.CustomAugmentation(self._transform, probability=1))
        gen.add(image.Resize(self.image_size))
        gen.add(image.RandomFlipLR(probability=0.5))
        gen.add(image.RandomColorAugmentors(probability=0.5))
        gen.add(image.RandomErasing(probability=0.5))
        gen.add(image.ProcessInput(self.get_preprocess_input(), batch_axis=True))
        if encode_truth:
            gen.add(image.ProcessOutput(lambda y: y if y is None else self.encode_truth([y])[0]))
        return gen

    def _transform(self, rgb: np.ndarray, y: ml.ObjectsAnnotation, w, rand: np.random.RandomState, ctx: generator.GeneratorContext):
        """変形を伴うAugmentation。"""
        assert ctx is not None
        aspect_rations = (3 / 4, 4 / 3)
        aspect_prob = 0.5
        ar = np.sqrt(rand.choice(aspect_rations)) if rand.rand() <= aspect_prob else 1
        ar_list = np.array([ar, 1 / ar])
        # padding or crop
        if rand.rand() <= 0.5:
            rgb = self._padding(rgb, y, rand, ar_list)
        else:
            rgb = self._crop(rgb, y, rand, ar_list)
        return rgb, y, w

    def _padding(self, rgb, y, rand, ar_list):
        """Padding(zoom-out)。"""
        old_size = np.array([rgb.shape[1], rgb.shape[0]])
        for _ in range(30):
            pr = np.exp(rand.uniform(np.log(1), np.log(4)))  # SSD風：[1, 16]
            padded_size = np.ceil(old_size * np.maximum(pr * ar_list, 1)).astype(int)
            padding_size = padded_size - old_size
            paste_xy = np.array([rand.randint(0, padding_size[0] + 1), rand.randint(0, padding_size[1] + 1)])
            bboxes = np.copy(y.bboxes)
            bboxes = (np.tile(paste_xy, 2) + bboxes * np.tile(old_size, 2)) / np.tile(padded_size, 2)
            sb = bboxes * np.tile(self.image_size, 2)
            if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいbboxが発生するのはNG
                continue
            y.bboxes = bboxes
            # 先に縮小
            new_size = np.floor(old_size * self.image_size / padded_size).astype(int)
            rgb = ndimage.resize(rgb, new_size[0], new_size[1], padding=None)
            # パディング
            paste_lr = np.floor(paste_xy * self.image_size / padded_size).astype(int)
            paste_tb = self.image_size - (paste_lr + new_size)
            padding = rand.choice(('edge', 'zero', 'one', 'rand'))
            rgb = ndimage.pad_ltrb(rgb, paste_lr[0], paste_lr[1], paste_tb[0], paste_tb[1], padding, rand)
            assert rgb.shape[:2] == self.image_size
            break
        return rgb

    def _crop(self, rgb, y, rand, ar_list):
        """Crop(zoom-in)。"""
        # SSDでは結構複雑なことをやっているが、とりあえず簡単に実装
        bb_center = ml.bboxes_center(y.bboxes)
        bb_area = ml.bboxes_area(y.bboxes)
        old_size = np.array([rgb.shape[1], rgb.shape[0]])
        for _ in range(30):
            cr = np.exp(rand.uniform(np.log(np.sqrt(0.1)), np.log(1)))  # SSD風：[0.1, 1]
            cropped_wh = np.floor(old_size * np.minimum(cr * ar_list, 1)).astype(int)
            cropping_size = old_size - cropped_wh
            crop_xy = np.array([rand.randint(0, cropping_size[0] + 1), rand.randint(0, cropping_size[1] + 1)])
            crop_box = np.concatenate([crop_xy, crop_xy + cropped_wh]) / np.tile(old_size, 2)
            # 中心を含むbboxのみ有効
            bb_mask = math.in_range(bb_center, crop_box[:2], crop_box[2:]).all(axis=-1)
            if not bb_mask.any():
                continue
            # あまり極端に面積が減っていないbboxのみ有効
            lt = np.maximum(crop_box[np.newaxis, :2], y.bboxes[:, :2])
            rb = np.minimum(crop_box[np.newaxis, 2:], y.bboxes[:, 2:])
            cropped_area = (rb - lt).prod(axis=-1) * (lt < rb).all(axis=-1)
            bb_mask = np.logical_and(bb_mask, cropped_area >= bb_area * 0.3)
            # bboxが一つも残らなければやり直し
            if not bb_mask.any():
                continue
            bboxes = np.copy(y.bboxes)
            bboxes = (bboxes * np.tile(old_size, 2) - np.tile(crop_xy, 2)) / np.tile(cropped_wh, 2)
            bboxes = np.clip(bboxes, 0, 1)
            sb = bboxes * np.tile(self.image_size, 2)
            if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいbboxが発生するのはNG
                continue
            y.bboxes = bboxes[bb_mask]
            y.classes = y.classes[bb_mask]
            y.difficults = y.difficults[bb_mask]
            # 切り抜き
            rgb = ndimage.crop(rgb, crop_xy[0], crop_xy[1], cropped_wh[0], cropped_wh[1])
            assert (rgb.shape[:2] == cropped_wh[::-1]).all()
            break
        return rgb

    def predict(self, model, X, conf_threshold=0.1, verbose=1):
        """予測。"""
        def _filter_pred(pred):
            pred_classes = pred[:, 0].astype(np.int32)
            pred_confs = pred[:, 1]
            pred_locs = pred[:, 2:]
            mask = pred_confs >= conf_threshold
            return pred_classes[mask], pred_confs[mask], pred_locs[mask, :]

        result_list = []
        steps = model.gen.steps_per_epoch(len(X), model.batch_size)
        with utils.tqdm(total=len(X), unit='f', desc='predict', disable=verbose == 0) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, X_batch in enumerate(model.gen.flow(X, batch_size=model.batch_size)):
                    pred_list = model.model.predict_on_batch(X_batch)
                    result_list.append(executor.map(_filter_pred, pred_list))
                    # 次へ
                    pbar.update(len(X_batch))
                    if i + 1 >= steps:
                        assert i + 1 == steps
                        break
            all_results = sum([list(r) for r in result_list], [])
            pred_classes_list, pred_confs_list, pred_locs_list = zip(*all_results)

        return pred_classes_list, pred_confs_list, pred_locs_list
