import pytoolkit as tk

from ._core import Model


class KerasModel(Model):
    """Kerasのモデル。

    Args:
        create_model_fn (callable): モデルを作成する関数。
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        batch_size (int): バッチサイズ
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists (bool): モデルが存在してもスキップせず再学習するならFalse。
        fit_params (dict): tk.models.fit()のパラメータ

    """

    def __init__(
        self,
        create_model_fn,
        train_preprocessor=None,
        val_preprocessor=None,
        batch_size=32,
        *,
        model_name_format="model.fold{fold}.h5",
        skip_if_exists=True,
        load_model_fn=None,
        use_horovod=False,
        fit_params=None,
    ):
        self.create_model_fn = create_model_fn
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.batch_size = batch_size
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.fit_params = fit_params
        self.load_model_fn = load_model_fn
        self.use_horovod = use_horovod
        self.models_dir_ = None  # TODO: TF2.0でsession廃止してリファクタリング
        self.nfold_ = None  # TODO: TF2.0でsession廃止してリファクタリング

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (None未対応)

        Returns:
            dict: metrics名と値

        """
        tk.training.cv(
            create_model_fn=self.create_model_fn,
            train_dataset=dataset,
            folds=folds,
            train_preprocessor=self.train_preprocessor,
            val_preprocessor=self.val_preprocessor,
            batch_size=self.batch_size,
            models_dir=models_dir,
            model_name_format=self.model_name_format,
            skip_if_exists=self.skip_if_exists,
            **(self.fit_params or {}),
        )
        return {}  # TODO: score

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        self.models_dir_ = models_dir
        # foldを数える (仮)
        for fold in range(999):
            path = models_dir / self.model_name_format.format(fold=fold)
            if path.exists():
                self.nfold_ = fold + 1
            else:
                break

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        return tk.training.predict_cv(
            dataset=dataset,
            nfold=self.nfold_,
            preprocessor=self.val_preprocessor,
            batch_size=self.batch_size,  # TODO: 2倍？
            load_model_fn=self.load_model_fn,
            use_horovod=self.use_horovod,
            models_dir=self.models_dir_,
            oof=False,
        )
