import pathlib

import numpy as np

import pytoolkit as tk

from ._core import Model


class KerasModel(Model):
    """Kerasのモデル。

    _saveではなく学習時にsaveしてしまう。

    Args:
        create_model_fn (callable): モデルを作成する関数。
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        batch_size (int): バッチサイズ
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists (bool): モデルが存在してもスキップせず再学習するならFalse。
        fit_params (dict): tk.models.fit()のパラメータ
        load_model_fn (callable): モデルを読み込む関数
        use_horovod (bool): MPIによる分散処理をするか否か。
        models_dir (PathLike): 保存先ディレクトリ

    """

    def __init__(
        self,
        create_model_fn,
        train_preprocessor,
        val_preprocessor,
        batch_size=32,
        *,
        models_dir,
        model_name_format="model.fold{fold}.h5",
        skip_if_exists=True,
        fit_params=None,
        load_model_fn=None,
        use_horovod=False,
        preprocessors=None,
        postprocessors=None,
    ):
        super().__init__(preprocessors, postprocessors)
        self.create_model_fn = create_model_fn
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.batch_size = batch_size
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.fit_params = fit_params
        self.load_model_fn = load_model_fn
        self.use_horovod = use_horovod
        self.models_dir = pathlib.Path(models_dir)
        self.models = None

    def _save(self, models_dir):
        assert models_dir == self.models_dir

    def _load(self, models_dir):
        assert models_dir == self.models_dir
        load_model_fn = self.load_model_fn or tk.models.load
        self.models = []
        for fold in range(999):
            model_path = models_dir / self.model_name_format.format(fold=fold)
            if model_path.exists():
                self.models.append(load_model_fn(model_path))
            else:
                break

    def _cv(self, dataset, folds):
        self.models = []
        score_list = []
        score_weights = []
        for fold, (train_indices, val_indices) in enumerate(folds):
            tk.log.get(__name__).info(
                f"Fold {fold + 1}/{len(folds)}: train={len(train_indices)} val={len(val_indices)}"
            )
            train_dataset = dataset.slice(train_indices)
            val_dataset = dataset.slice(val_indices)
            model = self.create_model_fn()
            tk.models.fit(
                model,
                train_dataset=train_dataset,
                train_preprocessor=self.train_preprocessor,
                val_dataset=val_dataset,
                val_preprocessor=self.val_preprocessor,
                batch_size=self.batch_size,
                **(self.fit_params or {}),
            )
            model_path = self.models_dir / self.model_name_format.format(fold=fold)
            tk.models.save(model, model_path)
            self.models.append(model)
            scores = tk.models.evaluate(
                model,
                val_dataset,
                preprocessor=self.val_preprocessor,
                batch_size=self.batch_size,
                use_horovod=True,
            )
            for k, v in scores.items():
                tk.log.get(__name__).info(
                    f"Fold {fold + 1}/{len(folds)}: val_{k}={v:,.3f}"
                )
            score_list.append(scores)
            score_weights.append(len(val_indices))

        return {
            k: np.average([s[k] for s in score_list], weights=score_weights)
            for k in score_list[0]
        }

    def _predict(self, dataset):
        return [
            tk.models.predict(
                model,
                dataset,
                self.val_preprocessor,
                self.batch_size,
                desc=f"fold",
                use_horovod=self.use_horovod,
                # TODO: TTA
            )
            for model in self.models
        ]
