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


def create_generator(image_size, keep_aspect, preprocess_input, encode_truth, flip_h, flip_v, rotate90):
    """ImageDataGeneratorを作って返す。"""
    gen = image.ImageDataGenerator()
    gen.add(image.RandomAlpha(probability=0.5))
    gen.add(image.RandomZoom(probability=1, output_size=image_size, keep_aspect=keep_aspect))
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
        gen.add(generator.ProcessOutput(lambda y: encode_truth([y])[0]))
    return gen
