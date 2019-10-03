from . import (
    applications,
    autoaugment,
    backend,
    cache,
    callbacks,
    cli,
    data,
    datasets,
    dl,
    evaluations,
    hpo,
    hvd,
    image,
    layers,
    log,
    losses,
    math,
    metrics,
    ml,
    models,
    ndimage,
    notifications,
    od,
    optimizers,
    pipeline,
    preprocessing,
    table,
    threading,
    typing,
    utils,
    validation,
    vis,
    web,
)


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    custom_objects = {}
    custom_objects.update(layers.get_custom_objects())
    custom_objects.update(optimizers.get_custom_objects())
    return custom_objects
