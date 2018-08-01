"""お手製Object detection。

https://github.com/ak110/object_detector
"""
import pathlib
import typing

import numpy as np

from . import hvd, models, od_gen, od_net, od_pb
from .. import jsonex, log, ml, utils

# バージョン
_JSON_VERSION = '0.0.2'
# PASCAL VOC 07+12 trainvalで学習したときのmodel.json
_VOC_JSON_DATA = {
    "input_size": [320, 320],
    "map_sizes": [40, 20, 10],
    "num_classes": 20,
    "pb_size_patterns": [
        [1.2137997150421143, 1.6073973178863525],
        [2.3104209899902344, 3.817656993865967],
        [3.239943265914917, 7.371336936950684],
        [5.479208946228027, 4.491170406341553],
        [4.76220703125, 11.572043418884277],
        [10.89842414855957, 5.794517517089844],
        [7.733386993408203, 8.85787582397461],
        [4.474754333496094, 16.686479568481445]
    ],
    "version": "0.0.2"
}
# PASCAL VOC 07+12 trainvalで学習したときの重みファイル
_VOC_WEIGHTS_320_NAME = 'pytoolkit_od_voc_320.h5'
_VOC_WEIGHTS_320_URL = 'https://github.com/ak110/object_detector/releases/download/v0.0.2/model.320.h5'
_VOC_WEIGHTS_320_MD5 = 'a76081cb833dd301a381166ee14d574f'
_VOC_WEIGHTS_640_NAME = 'pytoolkit_od_voc_640.h5'
_VOC_WEIGHTS_640_URL = 'https://github.com/ak110/object_detector/releases/download/v0.0.2/model.640.h5'
_VOC_WEIGHTS_640_MD5 = '2b6f424938267634d5a34b43afb54a1b'


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    def __init__(self, input_size, map_sizes, num_classes):
        self.pb = od_pb.PriorBoxes(input_size, map_sizes, num_classes)
        self.model: models.Model = None

    def save(self, path: typing.Union[str, pathlib.Path]):
        """保存。"""
        data = {
            'version': _JSON_VERSION,
            'input_size': self.pb.input_size,
            'map_sizes': self.pb.map_sizes,
            'num_classes': self.pb.num_classes,
        }
        data.update(self.pb.to_dict())
        jsonex.dump(data, path)

    @staticmethod
    def load(path: typing.Union[str, pathlib.Path]):
        """読み込み。(ファイル)"""
        return ObjectDetector.load_from_dict(jsonex.load(path))

    @staticmethod
    def load_from_dict(data: dict):
        """読み込み。(dict)"""
        if data['version'] == '0.0.1':
            data.update(data.pop('pb'))
        od = ObjectDetector(
            input_size=data.get('input_size'),
            map_sizes=data.get('map_sizes'),
            num_classes=data.get('num_classes'))
        od.pb.from_dict(data)
        return od

    @staticmethod
    def load_voc(batch_size, input_size=(320, 320), keep_aspect=False, strict_nms=True, use_multi_gpu=True):
        """PASCAL VOC 07+12 trainvalで学習済みのモデルを読み込む。

        # 引数
        - batch_size: 予測時のバッチサイズ。
        - keep_aspect: padding / cropの際にアスペクト比を保持するならTrue、正方形にリサイズしてしまうならFalse。
        - strict_nms: クラスによらずNon-maximum suppressionするならTrue。(mAPは下がるが、重複したワクが出ないので実用上は良いはず)
        - use_multi_gpu: 予測をマルチGPUで行うならTrue。

        """
        assert input_size in ((320, 320), (640, 640))
        data = _VOC_JSON_DATA.copy()
        data['input_size'] = input_size
        od = ObjectDetector.load_from_dict(data)
        od.load_weights(weights='voc', batch_size=batch_size, keep_aspect=keep_aspect,
                        strict_nms=strict_nms, use_multi_gpu=use_multi_gpu)
        return od

    def fit(self, X_train: [pathlib.Path], y_train: [ml.ObjectsAnnotation],
            X_val: [pathlib.Path], y_val: [ml.ObjectsAnnotation],
            batch_size, epochs, lr_scale=1,
            initial_weights='voc', pb_size_pattern_count=8,
            flip_h=True, flip_v=False, rotate90=False,
            padding_rate=16, crop_rate=0.1, keep_aspect=False,
            aspect_prob=0.5, max_aspect_ratio=3 / 2, min_object_px=4,
            plot_path=None, tsv_log_path=None):
        """学習。

        # 引数
        - lr_scale: 学習率を調整するときの係数
        - initial_weights: 重みの初期値。
                           'imagenet'ならバックボーンのみ。
                           'voc'ならPASCAL VOC 07+12 trainvalで学習済みのもの。
                           ファイルパスならそれを読む。
        - pb_size_pattern_count: Prior boxのサイズ・アスペクト比のパターンの種類数。
        - flip_h: Data augmentationで水平flipを行うか否か。
        - flip_v: Data augmentationで垂直flipを行うか否か。
        - rotate90: Data augmentationで0, 90, 180, 270度の回転を行うか否か。
        - padding_rate: paddingする場合の面積の比の最大値。16なら最大で縦横4倍。
        - crop_rate: cropする場合の面積の比の最大値。0.1なら最小で縦横0.32倍。
        - keep_aspect: padding / cropの際にアスペクト比を保持するならTrue、正方形にリサイズしてしまうならFalse。
        - aspect_prob: アスペクト比を歪ませる確率。
        - max_aspect_ratio: アスペクト比を最大どこまで歪ませるか。(1.5なら正方形から3:2までランダムに歪ませる)
        - min_object_px: paddingなどでどこまでオブジェクトが小さくなるのを許容するか。(ピクセル数)
        - plot_path: ネットワークの図を出力するならそのパス。拡張子はpngやsvgなど。
        - tsv_log_path: lossなどをtsvファイルに出力するならそのパス。
        """
        assert self.model is None
        assert lr_scale > 0
        # 訓練データに合わせたprior boxの作成
        if hvd.is_master():
            self.pb.fit(y_train, pb_size_pattern_count, rotate90=rotate90, keep_aspect=keep_aspect)
            pb_dict = self.pb.to_dict()
        else:
            pb_dict = None
        pb_dict = hvd.bcast(pb_dict)
        self.pb.from_dict(pb_dict)
        # prior boxのチェック
        if hvd.is_master():
            self.pb.summary()
            if y_val is not None:
                self.pb.check_prior_boxes(y_val)
        hvd.barrier()
        # モデルの作成
        network, lr_multipliers = od_net.create_network(pb=self.pb, mode='train', strict_nms=None)
        pi = od_net.get_preprocess_input()
        gen = od_gen.create_generator(self.pb.input_size, pi, self.pb.encode_truth,
                                      flip_h=flip_h, flip_v=flip_v, rotate90=rotate90,
                                      padding_rate=padding_rate, crop_rate=crop_rate, keep_aspect=keep_aspect,
                                      aspect_prob=aspect_prob, max_aspect_ratio=max_aspect_ratio,
                                      min_object_px=min_object_px)
        self.model = models.Model(network, gen, batch_size)
        self.model.summary()
        if plot_path:
            self.model.plot(plot_path)
        # 重みの読み込み
        logger = log.get(__name__)
        if initial_weights == 'imagenet':
            warm_start = False  # cold start
        else:
            if initial_weights == 'voc':
                initial_weights = self._get_voc_weights()
            else:
                initial_weights = pathlib.Path(initial_weights)
            self.model.load_weights(initial_weights, strict_warnings=False)
            logger.info(f'warm start: {initial_weights.name}')
            warm_start = True
        # 学習
        if warm_start:
            sgd_lr = lr_scale * 0.5 / 256 / 6 / 2  # lossが複雑なので微調整
            self.model.compile(sgd_lr=sgd_lr, lr_multipliers=None, loss=self.pb.loss, metrics=self.pb.metrics)
        else:
            sgd_lr = lr_scale * 0.5 / 256 / 6  # lossが複雑なので微調整
            self.model.compile(sgd_lr=sgd_lr, lr_multipliers=lr_multipliers, loss=self.pb.loss, metrics=self.pb.metrics)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, tsv_log_path=tsv_log_path,
                       cosine_annealing=True)

    def save_weights(self, path: typing.Union[str, pathlib.Path]):
        """重みの保存。(学習後用)"""
        assert self.model is not None
        self.model.save(path)

    def load_weights(self, weights: typing.Union[str, pathlib.Path], batch_size,
                     keep_aspect=False, strict_nms=True, use_multi_gpu=True):
        """重みの読み込み。(予測用)

        # 引数
        - weights: 読み込む重み。'voc'ならVOC07+12で学習したものを読み込む。pathlib.Pathならそのまま読み込む。
        - batch_size: 予測時のバッチサイズ。
        - keep_aspect: padding / cropの際にアスペクト比を保持するならTrue、正方形にリサイズしてしまうならFalse。
        - strict_nms: クラスによらずNon-maximum suppressionするならTrue。(mAPは下がるが、重複したワクが出ないので実用上は良いはず)
        - use_multi_gpu: 予測をマルチGPUで行うならTrue。

        """
        if self.model is not None:
            del self.model
        network, _ = od_net.create_network(pb=self.pb, mode='predict', strict_nms=strict_nms)
        pi = od_net.get_preprocess_input()
        gen = od_gen.create_predict_generator(self.pb.input_size, pi, keep_aspect=keep_aspect)
        self.model = models.Model(network, gen, batch_size)
        logger = log.get(__name__)
        if weights == 'voc':
            weights = self._get_voc_weights()
        else:
            weights = pathlib.Path(weights)
        self.model.load_weights(weights, strict_warnings=False)
        logger.info(f'{weights.name} loaded.')
        # マルチGPU化。
        if use_multi_gpu:
            gpus = utils.get_gpu_count()
            self.model.set_multi_gpu_model(gpus)
        else:
            gpus = 1
        # 1回予測して計算グラフを構築
        self.model.model.predict_on_batch(np.zeros((gpus,) + tuple(self.pb.input_size) + (3,), np.float32))
        logger.info('trainable params: %d', models.count_trainable_params(network))

    def predict(self, X, conf_threshold=0.01, verbose=1) -> [ml.ObjectsPrediction]:
        """予測。"""
        assert self.model is not None
        pred = []
        # ややトリッキーだが、パディングなどに備えて画像全体のboxをyとして与える。
        y = np.array([ml.ObjectsAnnotation('.', 300, 300, [0], [[0, 0, 1, 1]]) for _ in range(len(X))])
        g, steps = self.model.gen.flow(X, y, batch_size=self.model.batch_size)
        with utils.tqdm(total=len(X), unit='f', desc='predict', disable=verbose == 0) as pbar:
            for i, (X_batch, y_batch) in enumerate(g):
                # 予測
                pred_list = self.model.model.predict_on_batch(X_batch)
                # 整形：キャストしたりマスクしたり
                for yp, p in zip(y_batch, pred_list):
                    offset = np.tile(yp.bboxes[0, :2], (1, 2))
                    size = np.tile(yp.bboxes[0, 2:] - yp.bboxes[0, :2], (1, 2))
                    pred_classes = p[:, 0].astype(np.int32)
                    pred_confs = p[:, 1]
                    pred_locs = p[:, 2:]
                    pred_locs = (pred_locs - offset) / size  # パディング分の補正
                    mask = pred_confs >= conf_threshold
                    pred.append(ml.ObjectsPrediction(pred_classes[mask], pred_confs[mask], pred_locs[mask, :]))
                # 次へ
                pbar.update(len(X_batch))
                if i + 1 >= steps:
                    assert i + 1 == steps
                    break
        return pred

    def _get_voc_weights(self) -> pathlib.Path:
        """PASCAL VOCの学習済み重みのパスを返す。"""
        downsampling_count = max(self.pb.input_size) // self.pb.map_sizes[0]
        if downsampling_count <= 320 // 40:
            weights = hvd.get_file(
                _VOC_WEIGHTS_320_NAME, _VOC_WEIGHTS_320_URL,
                file_hash=_VOC_WEIGHTS_320_MD5, cache_subdir='models')
        else:
            weights = hvd.get_file(
                _VOC_WEIGHTS_640_NAME, _VOC_WEIGHTS_640_URL,
                file_hash=_VOC_WEIGHTS_640_MD5, cache_subdir='models')
        return weights
