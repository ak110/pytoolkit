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
    parser.add_argument('--input-size', default=320, type=int, nargs=2)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    train_names = tk.io.read_all_lines(args.vocdevkit_dir / 'VOC2012' / 'ImageSets' / 'Segmentation' / 'train.txt')
    val_names = tk.io.read_all_lines(args.vocdevkit_dir / 'VOC2012' / 'ImageSets' / 'Segmentation' / 'val.txt')
    X_train = np.array([args.vocdevkit_dir / 'VOC2012' / 'JPEGImages' / f'{n}.jpg' for n in train_names])
    y_train = np.array([args.vocdevkit_dir / 'VOC2012' / 'SegmentationClass' / f'{n}.png' for n in train_names])
    X_val = np.array([args.vocdevkit_dir / 'VOC2012' / 'JPEGImages' / f'{n}.jpg' for n in val_names])
    y_val = np.array([args.vocdevkit_dir / 'VOC2012' / 'SegmentationClass' / f'{n}.png' for n in val_names])

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
              mixup=True, cosine_annealing=True)
    model.save(args.result_dir / 'model.h5')


@tk.log.trace()
def _validate(args, X_val, y_val):
    model = tk.dl.ss.SemanticSegmentor.load(args.result_dir / 'model.h5')
    pred_val = model.predict(X_val)
    ious, miou = model.compute_mean_iou(y_val, pred_val)
    logger = tk.log.get(__name__)
    logger.info(f'mIoU={miou:.3f}')
    class_names = class_names = ('bg', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor')
    for cn, iou in zip(class_names, ious):
        logger.info(f'{cn:12s} IoU={iou:.3f}')


if __name__ == '__main__':
    _main()
