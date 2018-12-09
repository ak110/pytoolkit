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

        bboxes, classes, areas, crowdeds = [], [], [], []
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
                areas.append(obj['area'])
                crowdeds.append(obj['iscrowd'])

        annotations.append(ml.ObjectsAnnotation(
            path=coco_dir / dirname / filename,
            width=width,
            height=height,
            classes=classes,
            bboxes=bboxes,
            areas=areas,
            crowdeds=crowdeds))

    return np.array([t.path for t in annotations]), np.array(annotations), class_names


def evaluate(y_true, y_pred):
    """MS COCO向けのスコア算出。

    ChainerCVを利用。
    https://chainercv.readthedocs.io/en/stable/reference/evaluations.html?highlight=eval_detection_coco#chainercv.evaluations.eval_detection_coco

    # 戻り値
    - dict:
      - "map/iou=0.50:0.95/area=all/max_dets=100"
      - "map/iou=0.50/area=all/max_dets=100"
      - "map/iou=0.75/area=all/max_dets=100"
      - などなど

    """
    import chainercv
    gt_classes_list = np.array([y.classes for y in y_true])
    gt_bboxes_list = np.array([y.real_bboxes for y in y_true])
    gt_areas_list = np.array([y.areas for y in y_true])
    gt_crowdeds_list = np.array([y.crowdeds for y in y_true])
    pred_classes_list = np.array([p.classes for p in y_pred])
    pred_confs_list = np.array([p.confs for p in y_pred])
    pred_bboxes_list = np.array([p.get_real_bboxes(y.width, y.height) for (p, y) in zip(y_pred, y_true)])
    return chainercv.evaluations.eval_detection_coco(
        pred_bboxes_list, pred_classes_list, pred_confs_list,
        gt_bboxes_list, gt_classes_list, gt_areas_list, gt_crowdeds_list)
