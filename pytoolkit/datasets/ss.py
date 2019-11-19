"""セマンティックセグメンテーション関連。"""
from __future__ import annotations

import pathlib
import typing

import numpy as np

import pytoolkit as tk

# Cityscapes colors: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
_cityscapes_class_colors = np.array(
    [
        (128, 64, 128),  # flat         : road
        (244, 35, 232),  # flat         : sidewalk
        (70, 70, 70),  # construction : building
        (102, 102, 156),  # construction : wall
        (190, 153, 153),  # construction : fence
        (153, 153, 153),  # object       : pole
        (250, 170, 30),  # object       : traffic light
        (220, 220, 0),  # object       : traffic sign
        (107, 142, 35),  # nature       : vegetation
        (152, 251, 152),  # nature       : terrain
        (70, 130, 180),  # sky          : sky
        (220, 20, 60),  # human        : person
        (255, 0, 0),  # human        : rider
        (0, 0, 142),  # vehicle      : car
        (0, 0, 70),  # vehicle      : truck
        (0, 60, 100),  # vehicle      : bus
        (0, 80, 100),  # vehicle      : train
        (0, 0, 230),  # vehicle      : motorcycle
        (119, 11, 32),  # vehicle      : bicycle
    ]
)
_cityscapes_void_colors = np.array(
    [
        (
            0,
            0,
            0,
        ),  # void         : unlabeled, ego vehicle, rectification border, out of roi, static
        (111, 74, 0),  # void         : dynamic
        (81, 0, 81),  # void         : ground
        (250, 170, 160),  # flat         : parking
        (230, 150, 140),  # flat         : rail track
        (180, 165, 180),  # construction : guard rail
        (150, 100, 100),  # construction : bridge
        (150, 120, 90),  # construction : tunnel
        (153, 153, 153),  # object       : polegroup
        (0, 0, 90),  # vehicle      : caravan
        (0, 0, 110),  # vehicle      : trailer
        (0, 0, 142),  # vehicle      : license plate
    ]
)


def load_cityscapes(
    data_dir: tk.typing.PathLike, mode: str = "fine"
) -> typing.Tuple[tk.data.Dataset, tk.data.Dataset]:
    """Cityscapes Dataset <https://www.cityscapes-dataset.com/>

    Args:
        mode: データの種類。"fine" or "coarse"。

    Dataset.metadata:
        - class_colors: 評価対象のクラスのRGB値。shape=(N, 3)
        - void_colors: 評価対象外のクラスのRGB値。shape=(M, 3)

    """
    assert mode in ("fine", "coarse")
    data_dir = pathlib.Path(data_dir)
    name = {"fine": "gtFine", "coarse": "gtCoarse"}[mode]

    X_dir = data_dir / "leftImg8bit"
    y_dir = data_dir / name

    X_train = np.array(sorted((X_dir / "train").glob("*/*.png")))
    X_val = np.array(sorted((X_dir / "val").glob("*/*.png")))
    y_train = np.array(
        [
            y_dir
            / str(x.relative_to(X_dir)).replace(
                "_leftImg8bit.png", f"_{name}_color.png"
            )
            for x in X_train
        ]
    )
    y_val = np.array(
        [
            y_dir
            / str(x.relative_to(X_dir)).replace(
                "_leftImg8bit.png", f"_{name}_color.png"
            )
            for x in X_val
        ]
    )
    metadata = {
        "class_colors": _cityscapes_class_colors,
        "void_colors": _cityscapes_void_colors,
    }

    train_set = tk.data.Dataset(X_train, y_train, metadata=metadata)
    val_set = tk.data.Dataset(X_val, y_val, metadata=metadata)
    return train_set, val_set
