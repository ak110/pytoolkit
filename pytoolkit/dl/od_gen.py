"""お手製Object detectionのGenerator。"""

from .. import generator, image


def create_pretrain_generator(image_size, preprocess_input):
    """ImageDataGeneratorを作って返す。"""
    gen = image.ImageDataGenerator()
    gen.add(image.Resize(image_size))
    gen.add(image.RandomFlipLR(probability=0.5))
    gen.add(image.RandomErasing(probability=0.5))
    gen.add(image.RotationsLearning())
    gen.add(generator.ProcessInput(preprocess_input, batch_axis=True))
    return gen


def create_generator(image_size, preprocess_input, encode_truth,
                     padding_rate=16, crop_rate=0.1, keep_aspect=False,
                     aspect_prob=0.5, max_aspect_ratio=3 / 2,
                     min_object_px=4,
                     flip_h=True, flip_v=False, rotate90=False):
    """ImageDataGeneratorを作って返す。"""
    gen = image.ImageDataGenerator()
    gen.add(image.RandomAlpha(probability=0.5))
    gen.add(image.RandomZoom(probability=1, output_size=image_size, keep_aspect=keep_aspect,
                             padding_rate=padding_rate, crop_rate=crop_rate,
                             aspect_prob=aspect_prob, max_aspect_ratio=max_aspect_ratio,
                             min_object_px=min_object_px))
    if flip_h:
        gen.add(image.RandomFlipLR(probability=0.5))
    if flip_v:
        gen.add(image.RandomFlipTB(probability=0.5))
    if rotate90:
        gen.add(image.RandomRotate90(probability=1))
    gen.add(image.RandomColorAugmentors())
    gen.add(image.RandomErasing(probability=0.5))
    gen.add(generator.ProcessInput(preprocess_input, batch_axis=True))
    if encode_truth is not None:
        gen.add(generator.ProcessOutput(lambda y: encode_truth([y])[0]))
    return gen
