"""お手製Object detectionのGenerator。"""

import numpy as np

from .. import generator, image, math, ml, ndimage


def create_pretrain_generator(image_size, preprocess_input):
    """ImageDataGeneratorを作って返す。"""
    gen = image.ImageDataGenerator()
    gen.add(image.Resize(image_size))
    gen.add(image.RandomFlipLR(probability=0.5))
    gen.add(image.RandomErasing(probability=0.5))
    gen.add(image.RotationsLearning())
    gen.add(generator.ProcessInput(preprocess_input, batch_axis=True))
    return gen


def create_generator(image_size, preprocess_input, encode_truth, flip_h, flip_v, rotate90):
    """ImageDataGeneratorを作って返す。"""
    def _transform(rgb: np.ndarray, y: ml.ObjectsAnnotation, w, rand: np.random.RandomState, ctx: generator.GeneratorContext):
        """変形を伴うAugmentation。"""
        assert ctx is not None
        aspect_rations = (3 / 4, 4 / 3)
        aspect_prob = 0.5
        ar = np.sqrt(rand.choice(aspect_rations)) if rand.rand() <= aspect_prob else 1
        ar_list = np.array([ar, 1 / ar])
        # padding or crop
        if rand.rand() <= 0.5:
            rgb = _padding(image_size, rgb, y, rand, ar_list)
        else:
            rgb = _crop(image_size, rgb, y, rand, ar_list)
        return rgb, y, w

    gen = image.ImageDataGenerator()
    gen.add(image.RandomAlpha(probability=0.5))
    gen.add(generator.CustomAugmentation(_transform, probability=1))
    gen.add(image.Resize(image_size))
    if flip_h:
        gen.add(image.RandomFlipLR(probability=0.5))
    if flip_v:
        gen.add(image.RandomFlipTB(probability=0.5))
    if rotate90:
        gen.add(image.RandomRotate90(probability=1))
    gen.add(image.RandomColorAugmentors(probability=0.5))
    gen.add(image.RandomErasing(probability=0.5))
    gen.add(generator.ProcessInput(preprocess_input, batch_axis=True))
    if encode_truth is not None:
        gen.add(generator.ProcessOutput(lambda y: y if y is None else encode_truth([y])[0]))
    return gen


def _padding(image_size, rgb, y, rand, ar_list):
    """Padding(zoom-out)。"""
    old_size = np.array([rgb.shape[1], rgb.shape[0]])
    for _ in range(30):
        pr = np.exp(rand.uniform(np.log(1), np.log(4)))  # SSD風：[1, 16]
        padded_size = np.ceil(old_size * np.maximum(pr * ar_list, 1)).astype(int)
        padding_size = padded_size - old_size
        paste_xy = np.array([rand.randint(0, padding_size[0] + 1), rand.randint(0, padding_size[1] + 1)])
        bboxes = np.copy(y.bboxes)
        bboxes = (np.tile(paste_xy, 2) + bboxes * np.tile(old_size, 2)) / np.tile(padded_size, 2)
        sb = bboxes * np.tile(image_size, 2)
        if (sb[:, 2:] - sb[:, :2] < 4).any():  # あまりに小さいbboxが発生するのはNG
            continue
        y.bboxes = bboxes
        # 先に縮小
        new_size = np.floor(old_size * image_size / padded_size).astype(int)
        rgb = ndimage.resize(rgb, new_size[0], new_size[1], padding=None)
        # パディング
        paste_lr = np.floor(paste_xy * image_size / padded_size).astype(int)
        paste_tb = image_size - (paste_lr + new_size)
        padding = rand.choice(('edge', 'zero', 'one', 'rand'))
        rgb = ndimage.pad_ltrb(rgb, paste_lr[0], paste_lr[1], paste_tb[0], paste_tb[1], padding, rand)
        assert rgb.shape[:2] == image_size
        break
    return rgb


def _crop(image_size, rgb, y, rand, ar_list):
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
        sb = bboxes * np.tile(image_size, 2)
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
