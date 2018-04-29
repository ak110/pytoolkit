"""お手製Object detection。

https://github.com/ak110/object_detector
"""
import pathlib
import typing

import numpy as np

from . import callbacks as dl_callbacks, hvd, models, od_gen, od_net, od_pb
from .. import jsonex, log, ml, utils

# バージョン
_JSON_VERSION = '0.0.1'


class ObjectDetector(object):
    """モデル。

    候補として最初に準備するboxの集合を持つ。
    """

    def __init__(self, base_network, input_size, map_sizes, num_classes):
        self.base_network = base_network
        self.input_size = tuple(input_size)
        self.pb = od_pb.PriorBoxes(map_sizes, num_classes)
        self.model: models.Model = None

    def save(self, path: typing.Union[str, pathlib.Path]):
        """保存。"""
        jsonex.dump({
            'version': _JSON_VERSION,
            'base_network': self.base_network,
            'input_size': self.input_size,
            'pb': self.pb.to_dict(),
            'num_classes': self.pb.num_classes,
        }, path)

    @staticmethod
    def load(path: typing.Union[str, pathlib.Path]):
        """読み込み。"""
        data = jsonex.load(path)
        od = ObjectDetector(
            base_network=data['base_network'],
            input_size=data['input_size'],
            map_sizes=data['pb']['map_sizes'],
            num_classes=data['num_classes'])
        od.pb.from_dict(data['pb'])
        return od

    def fit(self, X_train: [pathlib.Path], y_train: [ml.ObjectsAnnotation],
            X_val: [pathlib.Path], y_val: [ml.ObjectsAnnotation],
            batch_size, epochs, initial_weights=None, pb_size_pattern_count=8,
            flip_h=True, flip_v=False, rotate90=False,
            plot_path=None, history_path=None):
        """学習。"""
        assert self.model is None
        # 訓練データに合わせたprior boxの作成
        if hvd.is_master():
            logger = log.get(__name__)
            logger.info(f'base network:         {self.base_network}')
            logger.info(f'input size:           {self.input_size}')
            self.pb.fit(y_train, self.input_size, pb_size_pattern_count)
            pb_dict = self.pb.to_dict()
        else:
            pb_dict = None
        pb_dict = hvd.bcast(pb_dict)
        self.pb.from_dict(pb_dict)
        # prior boxのチェック
        if hvd.is_master():
            self.pb.summary()
            self.pb.check_prior_boxes(y_val)
        hvd.barrier()
        # モデルの作成
        self._create_model(mode='train', weights=initial_weights, batch_size=batch_size,
                           flip_h=flip_h, flip_v=flip_v, rotate90=rotate90)
        if plot_path:
            self.model.plot(plot_path)
        # 学習
        callbacks = []
        callbacks.append(dl_callbacks.learning_rate(reduce_epoch_rates=(0.5, 0.75, 0.875)))
        callbacks.extend(self.model.horovod_callbacks())
        if history_path:
            callbacks.append(dl_callbacks.tsv_logger(history_path))
        callbacks.append(dl_callbacks.epoch_logger())
        callbacks.append(dl_callbacks.freeze_bn(0.95))
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, callbacks=callbacks)

    def save_weights(self, path: typing.Union[str, pathlib.Path]):
        """重みの保存。(学習後用)"""
        assert self.model is not None
        self.model.save(path)

    def load_weights(self, path: typing.Union[str, pathlib.Path], batch_size, strict_nms=True, use_multi_gpu=True):
        """重みの読み込み。(予測用)"""
        assert self.model is None
        self._create_model(mode='predict', weights=path, batch_size=batch_size,
                           flip_h=False, flip_v=False, rotate90=False, strict_nms=strict_nms)
        # 予マルチGPU化。
        if use_multi_gpu:
            gpus = utils.get_gpu_count()
            self.model.set_multi_gpu_model(gpus)
        else:
            gpus = 1
        # 1回予測して計算グラフを構築
        self.model.model.predict_on_batch(np.zeros((gpus,) + self.input_size + (3,), np.float32))

    def predict(self, X, conf_threshold=0.1, verbose=1) -> [ml.ObjectsPrediction]:
        """予測。"""
        assert self.model is not None
        pred = []
        steps = self.model.gen.steps_per_epoch(len(X), self.model.batch_size)
        with utils.tqdm(total=len(X), unit='f', desc='predict', disable=verbose == 0) as pbar:
            for i, X_batch in enumerate(self.model.gen.flow(X, batch_size=self.model.batch_size)):
                # 予測
                pred_list = self.model.model.predict_on_batch(X_batch)
                # 整形：キャストしたりマスクしたり
                for p in pred_list:
                    pred_classes = p[:, 0].astype(np.int32)
                    pred_confs = p[:, 1]
                    pred_locs = p[:, 2:]
                    mask = pred_confs >= conf_threshold
                    pred.append(ml.ObjectsPrediction(pred_classes[mask], pred_confs[mask], pred_locs[mask, :]))
                # 次へ
                pbar.update(len(X_batch))
                if i + 1 >= steps:
                    assert i + 1 == steps
                    break
        return pred

    @log.trace()
    def _create_model(self, mode, weights, batch_size, flip_h, flip_v, rotate90, strict_nms=None):
        """学習とか予測とか用に`tk.dl.models.Model`を作成して返す。

        # 引数
        - mode: 'pretrain', 'train', 'predict'のいずれか。(出力などが違う)
        - weights: 読み込む重み。Noneなら読み込まない。'imagenet'ならバックボーンだけ。'voc'ならVOC07+12で学習したものを読み込む。pathlib.Pathならそのまま読み込む。

        """
        logger = log.get(__name__)

        network, lr_multipliers = od_net.create_network(
            base_network=self.base_network, input_size=self.input_size, pb=self.pb,
            mode=mode, strict_nms=strict_nms)
        pi = od_net.get_preprocess_input(self.base_network)
        if mode == 'pretrain':
            gen = od_gen.create_pretrain_generator(self.input_size, pi)
        else:
            gen = od_gen.create_generator(self.input_size, pi, self.pb.encode_truth,
                                          flip_h=flip_h, flip_v=flip_v, rotate90=rotate90)
        self.model = models.Model(network, gen, batch_size)
        if mode in ('pretrain', 'train'):
            self.model.summary()

        if weights == 'voc':
            pass  # TODO: githubに学習済みモデル置いてkeras.applicationsみたいなダウンロード機能作る。
        elif isinstance(weights, pathlib.Path):
            self.model.load_weights(weights)
            if mode == 'predict':
                logger.info(f'{weights.name} loaded.')
            else:
                logger.info(f'warm start: {weights.name}')
        else:
            logger.info(f'cold start.')

        if mode == 'pretrain':
            # 事前学習：通常の分類としてコンパイル
            self.model.compile(sgd_lr=0.5 / 256, loss='categorical_crossentropy', metrics=['acc'])
        elif mode == 'train':
            # Object detectionとしてコンパイル
            sgd_lr = 0.5 / 256 / 3  # lossが複雑なので微調整
            self.model.compile(sgd_lr=sgd_lr, lr_multipliers=lr_multipliers, loss=self.pb.loss, metrics=self.pb.metrics)
        else:
            assert mode == 'predict'
