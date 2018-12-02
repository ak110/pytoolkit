"""MS COCOデータセット関連。

以下の3ファイルを解凍した結果を格納しているディレクトリのパスを受け取って色々処理をする。

- [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
- [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

"""
import pathlib

import numpy as np

from .. import ml


def load_od(coco_dir, year=2017):
    """物体検出のデータを読み込む。

    # 引数
    - coco_dir: annotationsディレクトリなどが入っているディレクトリのパス
    - year: 読み込むデータの西暦

    # 戻り値
    - (X_train, y_train)
    - (X_val, y_val)
    - class_names
    """
    X_train, y_train, class_names = load_od_data(coco_dir, f'train{year}')
    X_val, y_val, class_names_val = load_od_data(coco_dir, f'val{year}')
    assert class_names == class_names_val
    return (X_train, y_train), (X_val, y_val), class_names


def load_od_data(coco_dir, data_name):
    """物体検出のデータの読み込み。"""
    from pycocotools.coco import COCO
    coco_dir = pathlib.Path(coco_dir)
    coco = COCO(str(coco_dir / 'annotations' / f'instances_{data_name}.json'))

    class_names = [c['name'] for c in coco.loadCats(coco.getCatIds())]
    jsonclassid_to_index = {c['id']: class_names.index(c['name']) for c in coco.loadCats(coco.getCatIds())}

    annotations = []
    for entry in coco.loadImgs(coco.getImgIds()):
        dirname, filename = entry['coco_url'].split('/')[-2:]
        objs = coco.loadAnns(coco.getAnnIds(imgIds=entry['id'], iscrowd=None))

        bboxes, classes = [], []
        width, height = entry['width'], entry['height']
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            x, y, w, h = obj['bbox']
            bbox = np.array([x, y, x + w, y + h]) / np.array([width, height, width, height])
            bbox = np.clip(bbox, 0, 1)
            if (bbox[:2] < bbox[2:]).all():
                bboxes.append(bbox)
                classes.append(jsonclassid_to_index[obj['category_id']])

        annotations.append(ml.ObjectsAnnotation(
            path=coco_dir / dirname / filename,
            width=width,
            height=height,
            classes=classes,
            bboxes=bboxes))

    return np.array([t.path for t in annotations]), np.array(annotations), class_names
