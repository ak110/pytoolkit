#!/usr/bin/env python3
"""実験用コード：MS COCOで物体検出の学習。"""
import argparse
import pathlib

import pytoolkit as tk


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='all', choices=('all', 'train', 'validate'), nargs='?')
    parser.add_argument('--coco-dir', default=pathlib.Path('data/coco'), type=pathlib.Path)
    parser.add_argument('--result-dir', default=pathlib.Path('results_voc'), type=pathlib.Path)
    parser.add_argument('--input-size', default=(320, 320), type=int, nargs=2)
    parser.add_argument('--map-sizes', default=(40, 20, 10), type=int, nargs='+')
    parser.add_argument('--pb-sizes', default=8, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--base-model', default=None, type=pathlib.Path)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    (X_train, y_train), (X_val, y_val), class_names = tk.data.coco.load_od(args.coco_dir)

    # 学習
    tk.dl.hvd.init()
    if args.mode in ('train', 'all'):
        with tk.dl.session(use_horovod=True):
            tk.log.init(args.result_dir / 'train.log')
            _train(args, X_train, y_train, X_val, y_val, class_names)

    # 検証
    if args.mode in ('validate', 'all'):
        if tk.dl.hvd.is_master():
            tk.log.init(args.result_dir / 'validate.log')
            with tk.dl.session():
                _validate(args, X_val, y_val, class_names)


@tk.log.trace()
def _train(args, X_train, y_train, X_val, y_val, class_names):
    num_classes = len(class_names)
    od = tk.dl.od.ObjectDetector(args.input_size, args.map_sizes, num_classes)
    od.fit(X_train, y_train, X_val, y_val,
           batch_size=args.batch_size, epochs=args.epochs,
           initial_weights='imagenet' if args.base_model is None else args.base_model,
           pb_size_pattern_count=args.pb_sizes,
           flip_h=True, flip_v=False, rotate90=False,
           plot_path=args.result_dir / 'model.svg',
           tsv_log_path=args.result_dir / 'history.tsv')
    od.save(args.result_dir / 'model.json')
    od.save_weights(args.result_dir / 'model.h5')


@tk.log.trace()
def _validate(args, X_val, y_val, class_names):
    od = tk.dl.od.ObjectDetector.load(args.result_dir / 'model.json')
    od.load_weights(args.result_dir / 'model.h5', batch_size=args.batch_size, strict_nms=True, use_multi_gpu=True)
    pred_val = od.predict(X_val, conf_threshold=0.25)
    # 適合率・再現率などを算出・表示
    precisions, recalls, fscores, supports = tk.ml.compute_scores(y_val, pred_val, iou_threshold=0.5)
    tk.ml.print_scores(precisions, recalls, fscores, supports, class_names)
    # 先頭部分のみ可視化
    save_dir = args.result_dir / '___check'
    for x, p in zip(X_val[:64], pred_val[:64]):
        img = p.plot(x, class_names)
        tk.ndimage.save(save_dir / (x.stem + '.jpg'), img)
    # mAPを算出・表示
    od.load_weights(args.result_dir / 'model.h5', batch_size=args.batch_size, strict_nms=False, use_multi_gpu=True)
    pred_val = od.predict(X_val)
    map1 = tk.ml.compute_map(y_val, pred_val, use_voc2007_metric=False)
    map2 = tk.ml.compute_map(y_val, pred_val, use_voc2007_metric=True)
    logger = tk.log.get(__name__)
    logger.info(f'mAP={map1:.3f} mAP(VOC2007)={map2:.3f}')


if __name__ == '__main__':
    _main()
