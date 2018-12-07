#!/usr/bin/env python3
"""実験用コード：PASCAL VOC 2012でセマンティックセグメンテーション。"""
import argparse
import pathlib

import numpy as np

import pytoolkit as tk


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='all', choices=('all', 'train', 'validate'), nargs='?')
    parser.add_argument('--vocdevkit-dir', default=pathlib.Path('data/VOCdevkit'), type=pathlib.Path)
    parser.add_argument('--result-dir', default=pathlib.Path('results_ss_voc'), type=pathlib.Path)
    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--binary', action='store_true')
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    train_names = tk.io.read_all_lines(args.vocdevkit_dir / 'VOC2012' / 'ImageSets' / 'Segmentation' / 'train.txt')
    val_names = tk.io.read_all_lines(args.vocdevkit_dir / 'VOC2012' / 'ImageSets' / 'Segmentation' / 'val.txt')
    X_train = np.array([args.vocdevkit_dir / 'VOC2012' / 'JPEGImages' / f'{n}.jpg' for n in train_names])
    y_train = np.array([args.vocdevkit_dir / 'VOC2012' / 'SegmentationClass' / f'{n}.png' for n in train_names])
    X_val = np.array([args.vocdevkit_dir / 'VOC2012' / 'JPEGImages' / f'{n}.jpg' for n in val_names])
    y_val = np.array([args.vocdevkit_dir / 'VOC2012' / 'SegmentationClass' / f'{n}.png' for n in val_names])
    # バイナリマスク化 (メモリ上でやっちゃう)
    if args.binary:
        y_train = np.array([((tk.ndimage.load(y, grayscale=True) >= 1) * 255).astype(np.uint8) for y in y_train], dtype=object)
        y_val = np.array([((tk.ndimage.load(y, grayscale=True) >= 1) * 255).astype(np.uint8) for y in y_val], dtype=object)

    # 学習
    tk.dl.hvd.init()
    if args.mode in ('train', 'all'):
        with tk.dl.session(use_horovod=True):
            tk.log.init(args.result_dir / 'train.log')
            _train(args, X_train, y_train, X_val, y_val)

    # 検証
    if args.mode in ('validate', 'all'):
        if tk.dl.hvd.is_master():
            tk.log.init(args.result_dir / 'validate.log')
            with tk.dl.session():
                _validate(args, X_val, y_val)


@tk.log.trace()
def _train(args, X_train, y_train, X_val, y_val):
    if args.binary:
        class_colors, void_color = None, None
    else:
        class_colors = [
            (0, 0, 0),  # bg
            (128, 0, 0),  # aeroplane
            (0, 128, 0),  # bicycle
            (128, 128, 0),  # bird
            (0, 0, 128),  # boat
            (128, 0, 128),  # bottle
            (0, 128, 128),  # bus
            (128, 128, 128),  # car
            (64, 0, 0),  # cat
            (192, 0, 0),  # chair
            (64, 128, 0),  # cow
            (192, 128, 0),  # diningtable
            (64, 0, 128),  # dog
            (192, 0, 128),  # horse
            (64, 128, 128),  # motorbike
            (192, 128, 128),  # person
            (0, 64, 0),  # potted plant
            (128, 64, 0),  # sheep
            (0, 192, 0),  # sofa
            (128, 192, 0),  # train
            (0, 64, 128),  # tv/monitor
        ]
        void_color = (224, 224, 192)
    model = tk.dl.ss.SemanticSegmentor.create(
        class_colors, void_color, args.input_size,
        batch_size=args.batch_size, rotation_type='mirror')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs,
              tsv_log_path=args.result_dir / 'history.tsv',
              mixup=False, cosine_annealing=True)
    model.save(args.result_dir / 'model.h5')


@tk.log.trace()
def _validate(args, X_val, y_val):
    model = tk.dl.ss.SemanticSegmentor.load(args.result_dir / 'model.h5')
    pred_val = model.predict(X_val, verbose=1)
    # 描画
    plot_step = len(X_val) // 32
    for x, p in tk.tqdm(list(zip(X_val[::plot_step], pred_val[::plot_step])), desc='plot'):
        tk.ndimage.save(args.result_dir / 'pred_val' / f'{x.stem}.soft.png', model.plot_mask(x, p, color_mode='soft'))
        tk.ndimage.save(args.result_dir / 'pred_val' / f'{x.stem}.hard.png', model.plot_mask(x, p, color_mode='hard'))
    # 評価(mean IoU)
    ious, miou = model.compute_mean_iou(y_val, pred_val)
    logger = tk.log.get(__name__)
    logger.info(f'mIoU={miou:.3f}')
    if args.binary:
        class_names = ('bg', 'obj')
    else:
        class_names = ('bg', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor')
    for cn, iou in zip(class_names, ious):
        logger.info(f'{cn:12s} IoU={iou:.3f}')
    # 評価その2
    mious = model.compute_mean_iou_per_image(y_val, pred_val)
    tk.math.print_histgram(mious)


if __name__ == '__main__':
    _main()