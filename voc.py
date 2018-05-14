#!/usr/bin/env python3
"""実験用コード：PASCAL VOC 07+12で物体検出の学習。"""
import argparse
import pathlib

import pytoolkit as tk


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocdevkit-dir', default=pathlib.Path('data/VOCdevkit'), type=pathlib.Path)
    parser.add_argument('--result-dir', default=pathlib.Path('results_voc'), type=pathlib.Path)
    parser.add_argument('--network', default='current', choices=('current', 'experimental', 'experimental_large'))
    parser.add_argument('--input-size', default=(320, 320), type=int, nargs=2)
    parser.add_argument('--map-sizes', default=(40, 20, 10), type=int, nargs='+')
    parser.add_argument('--pb-sizes', default=8, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    X_train, y_train = tk.data.voc.load_0712_trainval(args.vocdevkit_dir)
    X_val, y_val = tk.data.voc.load_07_test(args.vocdevkit_dir)

    # 学習(model.h5が存在しない場合のみ。学習後に消さずに再実行した場合は検証だけする。)
    if not (args.result_dir / 'model.h5').is_file():
        with tk.dl.session(use_horovod=True):
            tk.log.init(args.result_dir / 'train.log')
            _train(args, X_train, y_train, X_val, y_val)

    # 検証
    if tk.dl.hvd.is_master():
        tk.log.init(args.result_dir / 'validate.log')
        with tk.dl.session():
            _evaluate(args, X_val, y_val)
        with tk.dl.session():
            _validate(args, X_val, y_val)


@tk.log.trace()
def _train(args, X_train, y_train, X_val, y_val):
    # 重みがあれば読み込む
    weights = 'imagenet'
    for warm_path in (args.result_dir / 'model.base.h5', args.result_dir / 'pretrain.model.h5'):
        if warm_path.is_file():
            weights = warm_path
            break

    # 学習
    num_classes = len(tk.data.voc.CLASS_NAMES)
    od = tk.dl.od.ObjectDetector(args.network, args.input_size, args.map_sizes, num_classes)
    od.fit(X_train, y_train, X_val, y_val,
           batch_size=args.batch_size, epochs=args.epochs, initial_weights=weights, pb_size_pattern_count=args.pb_sizes,
           flip_h=True, flip_v=False, rotate90=False,
           plot_path=args.result_dir / 'model.svg',
           tsv_log_path=args.result_dir / 'train.history.tsv')
    # 保存
    od.save(args.result_dir / 'model.json')
    od.save_weights(args.result_dir / 'model.h5')


@tk.log.trace()
def _evaluate(args, X_val, y_val):
    od = tk.dl.od.ObjectDetector.load(args.result_dir / 'model.json')
    od.load_weights(args.result_dir / 'model.h5', batch_size=args.batch_size, strict_nms=True, use_multi_gpu=True)
    pred = od.predict(X_val, conf_threshold=0.75)
    # 適合率・再現率などを算出・表示
    precisions, recalls, fscores, supports = tk.ml.compute_scores(y_val, pred, iou_threshold=0.5)
    tk.ml.print_scores(precisions, recalls, fscores, supports, tk.data.voc.CLASS_NAMES)
    # 先頭部分のみ可視化
    save_dir = args.result_dir / '___check'
    for x, p in zip(X_val[:64], pred[:64]):
        img = p.plot(x, tk.data.voc.CLASS_NAMES)
        tk.ndimage.save(save_dir / (x.stem + '.jpg'), img)


@tk.log.trace()
def _validate(args, X_val, y_val):
    od = tk.dl.od.ObjectDetector.load(args.result_dir / 'model.json')
    od.load_weights(args.result_dir / 'model.h5', batch_size=args.batch_size, strict_nms=False, use_multi_gpu=True)
    pred_val = od.predict(X_val)
    # mAPを算出・表示
    map1 = tk.ml.compute_map(y_val, pred_val, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(y_val, pred_val, use_voc2007_metric=True)
    logger = tk.log.get(__name__)
    logger.info(f'mAP={map1:.3f} mAP(VOC2007)={map2:.3f}')


if __name__ == '__main__':
    _main()
