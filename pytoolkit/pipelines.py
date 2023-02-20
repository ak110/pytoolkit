"""テーブルデータのパイプライン。

ここでいうパイプラインとは、pl.DataFrameを受け取り、pl.DataFrameを返すステップの連なり。
各ステップは0個以上の依存するステップを持ち、依存するステップの出力を結合したものを受け取って
当該ステップの処理を行う。

各ステップは名前+インスタンスで扱う方が直感的な設計だが、
冗長なコードを減らしつつ未使用のimportでlinterの警告が発生しないことを重視し、
クラスで扱うこととした。

各ステップはpytoolkit.pipelines.Stepクラスやその派生クラスを継承して実装する。
実行はpytoolkit.pipelines.Pipelineに実行するステップのクラスを指定する。

各ステップの結果はファイルにキャッシュされる。
ステップや依存先のステップが定義されたファイルの更新日時が新しくなっていれば自動的に再計算する。
(TODO: リファクタリングなどで意図せず再計算が発生する場合があるため、
非常に計算に時間がかかるステップを実装する場合は、更新日時のチェックを無効化して必要に応じて手動で削除する)

主にコンペでの利用を想定し、ステップの実行には"train"と"test"の二種類がある。
また、それに伴い以下のパターンで実装を補助する派生クラスがある。

- "train"と"test"で同じ処理をするステップ: TransformStep
- "train"で事前に学習や統計情報の取得などの処理を行い、
  その後は"train"と"test"で同じ処理をするステップ: FitTransformStep
- コンペなどで"train"と"test"のデータを全部まとめて処理するステップ: AllDataStep
- "train"で学習してモデルを保存してout-of-fold predictionsを出力し、
  "test"でモデルを読み込んで推論をするステップ: ModelStep
- "train"のみ実装されているステップ: TrainOnlyStep

Pipelines.fineがTrueか否かでも必要に応じてキャッシュが別管理となる。
HPOなどの時間がかかる処理はStepの中でself.fineがTrueの場合のみ行うよう実装し、
コンペの提出時などだけfine=Trueで実行する。

"""
import abc
import inspect
import logging
import os
import pathlib
import time
import typing

import joblib
import polars as pl

logger = logging.getLogger(__name__)

RunType = typing.Literal["train", "test"]
InvokeType = typing.Union[RunType, typing.Literal["all"]]


class Step(metaclass=abc.ABCMeta):
    """パイプラインの各ステップ。

    Attributes:
        module_path: ステップが定義されたファイルのパス
        name: ステップ名 (既定値: self.module_path.stem)
        depends_on: 依存先ステップ名の配列 (既定値: [])
        use_file_cache: 結果をファイルにキャッシュするのか否か (既定値: True)
        use_memory_cache: 結果をメモリにキャッシュするのか否か (既定値: True)
        has_fine_train: fine=Trueな場合にtrainで特別な処理を実装しているのか否か (既定値: False)
        has_fine_test: fine=Trueな場合にtestで特別な処理を実装しているのか否か (既定値: False)
        logger: ロガー (派生クラスで実装時に使う用)

    Examples:

        派生クラスでuse_file_cacheなどを変更したい場合は__init__をオーバーライドする。

        ::

            def __init__(self) -> None:
                super().__init__()
                self.depends_on = []
                self.use_file_cache = True
                self.use_memory_cache = True
                self.has_fine_train = False
                self.has_fine_test = False

    """

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.module_path: pathlib.Path = pathlib.Path(inspect.getfile(self.__class__))
        self.name: str = self.module_path.stem
        self.depends_on: list[str] = []
        self.use_file_cache: bool = True
        self.use_memory_cache: bool = True
        self.has_fine_train: bool = False
        self.has_fine_test: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.fine_refcount = 0

    def has_fine(self, run_type: RunType) -> bool:
        """fine=Trueな場合に特別な処理を実装しているのか否か"""
        return self.has_fine_train if run_type == "train" else self.has_fine_test

    @property
    def fine(self) -> bool:
        """高精度な学習・推論を行うのか否か"""
        assert self.pipeline is not None
        self.fine_refcount += 1
        return self.pipeline.fine

    @property
    def model_dir(self) -> pathlib.Path:
        """モデルを保存したりしたいときのディレクトリ"""
        assert self.pipeline is not None
        fine_suffix = "-fine" if self.pipeline.fine and self.has_fine("train") else ""
        return self.pipeline.models_dir / (self.name + fine_suffix)

    @abc.abstractmethod
    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理。Pipeline経由で呼び出される。"""

    def invoke(
        self,
        step_types: str | list[str],
        invoke_type: InvokeType | None = None,
        cache: typing.Literal["use", "ignore", "disable"] = "use",
    ) -> pl.DataFrame:
        """指定ステップの実行。

        Args:
            step_types: 実行するステップのクラス
            run_type: 実行するステップの種類
            cache: ignoreにするとキャッシュがあっても読み込まない、disableにすると保存もしない。(伝播はしない)

        Returns:
            実行結果

        """
        assert self.pipeline is not None
        return self.pipeline.invoke(step_types, invoke_type, cache)


class Pipeline:
    """パイプラインを管理するクラス。

    Args:
        models_dir: モデルやログの保存先ディレクトリ
        cache_dir: キャッシュの保存先ディレクトリ
        fine: 高精度な学習・推論を行うのか否か

    """

    def __init__(
        self,
        models_dir: str | os.PathLike[str],
        cache_dir: str | os.PathLike[str],
        fine: bool = False,
    ) -> None:
        self.models_dir = pathlib.Path(models_dir)
        self.cache_dir = pathlib.Path(cache_dir)
        self.fine = fine
        self.memory_cache: dict[tuple[str, RunType], pl.DataFrame] = {}
        self.steps: dict[str, Step] = {}
        self.run_type_stack: list[RunType] = []
        self.step_stack: list[Step] = []
        self.logfmt = (
            "%(asctime)s [%(levelname)-5s] %(message)s"
            " <%(name)s> %(filename)s:%(lineno)d"
        )

    def add(self, step: Step) -> None:
        """ステップの追加。"""
        assert step.name not in self.steps, f"duplicated step name: {step.name}"
        step.pipeline = self
        self.steps[step.name] = step

    def check(self) -> None:
        """依存関係のチェック"""
        for s in self.steps.values():
            for d in s.depends_on:
                if d not in self.steps:
                    raise ValueError(f"'{d}' is not defined. (depend from {s.name})")

    def invoke(
        self,
        steps: str | list[str] | Step | list[Step],
        invoke_type: InvokeType | None,
        cache: typing.Literal["use", "ignore", "disable"] = "use",
    ) -> pl.DataFrame:
        """ステップの実行。

        Args:
            steps: 実行するステップの名前 or インスタンス。複数指定可。
            invoke_type: 実行するステップの種類。省略時は現在実行中のステップ。
            cache: ignoreにするとキャッシュがあっても読み込まない、disableにすると保存もしない。(伝播はしない)

        Returns:
            処理結果

        """
        steps = self._instantiate(steps)
        if len(steps) <= 0:
            return pl.DataFrame()

        if invoke_type == "all":
            df_train = self.invoke(steps, "train", cache)
            df_test = self.invoke(steps, "test", cache)
            return pl.concat([df_train, df_test])

        if invoke_type is None:
            assert len(self.run_type_stack) > 0, "依存関係の最上位ではinvoke_typeは省略不可"
            invoke_type = self.run_type_stack[-1]

        self.run_type_stack.append(invoke_type)
        try:
            return pl.concat(
                [self._invoke(step, invoke_type, cache) for step in steps],
                how="horizontal",
            )
        finally:
            self.run_type_stack.pop()

    def _instantiate(self, steps: str | list[str] | Step | list[Step]) -> list[Step]:
        """Stepのインスタンスを返す。"""
        if not isinstance(steps, list):
            steps = [steps]  # type: ignore[assignment]
        assert isinstance(steps, list)
        return [s if isinstance(s, Step) else self.get_step(s) for s in steps]

    def get_step(self, step_name: str) -> Step:
        """Stepのインスタンスを返す。"""
        step = self.steps.get(step_name)
        if step is None:
            raise ValueError(f"'{step_name}' is not defined.")
        return step

    def _invoke(
        self,
        step: Step,
        run_type: RunType,
        cache: typing.Literal["use", "ignore", "disable"],
    ) -> pl.DataFrame:
        self.step_stack.append(step)
        try:
            run_name = self._run_name(step, run_type)
            # ファイルキャッシュにあれば読んで返す
            cache_path = self.cache_dir / step.name / f"{run_name}.arrow"
            if cache == "use" and step.use_file_cache:
                if self._is_cache_valid(step, run_type):
                    logger.info(f"'{step.name}/{run_name}' load cache: {cache_path}")
                    return pl.read_ipc(cache_path)

            # メモリキャッシュにあれば返す
            if step.use_memory_cache:
                result = self.memory_cache.get((step.name, run_type))
                if result is not None:
                    logger.info(f"'{step.name}/{run_name}' get memory cache")
                    return result

            # ステップの実行
            result = self._run_step(step, run_type, run_name)

            # メモリキャッシュに保存
            if step.use_memory_cache:
                self.memory_cache[(step.name, run_type)] = result

            # ファイルキャッシュに保存
            if cache in ("use", "ignore") and step.use_file_cache:
                logger.info(f"'{step.name}/{run_name}' save cache: {cache_path}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                result.write_ipc(cache_path)
            return result
        finally:
            self.step_stack.pop()

    def _is_cache_valid(
        self, step: Step, run_type: RunType, base_cache_time: float | None = None
    ) -> bool:
        """キャッシュの有効性チェック"""
        # キャッシュを使わない場合は何であってもOKということにする
        if not step.use_file_cache:
            return True
        # キャッシュが無ければ無効
        run_name = self._run_name(step, run_type)
        cache_path = self.cache_dir / step.name / f"{run_name}.arrow"
        if not cache_path.exists():
            return False
        # 有効期限の簡易チェック。ソースコードの方が新しければNG。
        # ソースコード上の依存関係とかまでは追えないので注意。
        cache_time = cache_path.stat().st_mtime
        module_time = step.module_path.stat().st_mtime
        if cache_time <= module_time:
            logger.warning(f"cache expired: {cache_path}")
            cache_path.unlink()
            return False
        # 依存元のキャッシュより依存先のキャッシュが新しければNG
        if base_cache_time is None:
            base_cache_time = cache_time  # 依存元が無い
        elif base_cache_time <= cache_time:
            return False  # 依存元が古い
        else:
            base_cache_time = cache_time  # より古い方を基準に以降をチェック
        # 依存先のいずれかが無効なら無効
        if any(
            not self._is_cache_valid(s, run_type, base_cache_time)
            for s in self._instantiate(step.depends_on)
        ):
            return False
        # ここまで来たらOK
        return True

    def _run_step(self, step: Step, run_type: RunType, run_name: str) -> pl.DataFrame:
        """ステップの実行"""
        # ログの設定
        log_path = step.model_dir / f"{run_type}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path, mode="w", encoding="utf-8", delay=True
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(self.logfmt))
        root_logger = logging.getLogger(None)
        root_logger.addHandler(file_handler)
        step.logger.info(f"'{step.name}/{run_name}' start")
        start_time = time.perf_counter()
        try:
            # 依存関係の実行
            df = self.invoke(step.depends_on, run_type)
            # ステップの実行
            step.fine_refcount = 0
            result = step.run(df, run_type)
            # 行数チェック
            assert (
                len(df) == len(result) or len(df) == 0 or len(result) == 0
            ), f"Rows error: {step.name}/{run_name}"
            # fineの参照回数チェック
            if step.has_fine(run_type) == (step.fine_refcount == 0):
                step.logger.fatal(
                    f"'{step.name}/{run_name}' fine refcount error: {run_type=}"
                    f" has_fine={step.has_fine(run_type)} ref={step.fine_refcount=}"
                )
        finally:
            # ログを戻す
            step.logger.info(
                f"'{step.name}/{run_name}'"
                f" done in {time.perf_counter() - start_time:.0f} s"
            )
            root_logger.removeHandler(file_handler)
        return result

    def _run_name(self, step: Step, run_type: RunType) -> str:
        """ステップの実行名を作成して返す"""
        if self.fine and step.has_fine(run_type):
            return f"{run_type}-fine"
        else:
            return run_type


class TransformStep(Step, metaclass=abc.ABCMeta):
    """データ変換ステップ。

    Examples:
        ::

            class Step(pytoolkit.pipelines.TransformStep):
                def __init__(self):
                    super().__init__()
                    self.depends_on = ["xxx"]

                def transform(self, df: pl.DataFrame) -> pl.DataFrame:
                    return df

    """

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        del run_type  # noqa
        return self.transform(df)

    @abc.abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """変換処理"""


class FitTransformStep(Step, metaclass=abc.ABCMeta):
    """特徴量作成ステップ。

    Examples:
        ::

            class Step(pytoolkit.pipelines.FitTransformStep):
                def __init__(self):
                    super().__init__()
                    self.depends_on = ["xxx"]

                def fit(self, df_train: pl.DataFrame) -> typing.Any:
                    transformer = {}
                    return transformer

                def transform(self, transformer, df: pl.DataFrame) -> pl.DataFrame:
                    return df

    """

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        save_path = self.transformer_save_path
        if run_type == "train":
            transformer = self.fit(df)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_transformer(transformer, save_path)
        else:
            transformer = self.load_transformer(save_path)
        return self.transform(transformer, df)

    @abc.abstractmethod
    def fit(self, df_train: pl.DataFrame) -> typing.Any:
        """訓練＆保存"""

    @abc.abstractmethod
    def transform(self, transformer, df: pl.DataFrame) -> pl.DataFrame:
        """変換処理"""

    @property
    def transformer_save_path(self) -> pathlib.Path:
        """保存先パス"""
        return self.model_dir / "transformer.pkl"

    def save_transformer(self, transformer, save_path: pathlib.Path) -> typing.Any:
        """変換用情報の保存"""
        logger.info(f"save transformer: {save_path}")
        joblib.dump(transformer, save_path)

    def load_transformer(self, save_path: pathlib.Path) -> typing.Any:
        """変換用情報の読み込み"""
        logger.info(f"load transformer: {save_path}")
        return joblib.load(save_path)


class AllDataStep(Step, metaclass=abc.ABCMeta):
    """trainで全データを使って処理するステップ。_is_test_というbool列を追加して結合したものがdf_allとして渡される。

    Examples:
        ::

            class Step(pytoolkit.pipelines.AllDataStep):
                def __init__(self):
                    super().__init__()
                    self.depends_on = ["xxx"]

                def fit_transform(self, df_all: pl.DataFrame) -> pl.DataFrame:
                    return df_all

    """

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        if run_type == "train":
            df_test = self.invoke(self.depends_on, "test")
            df_all = pl.concat(
                [
                    df.with_columns(pl.lit(False).alias(self.is_test_column_name)),
                    df_test.with_columns(pl.lit(True).alias(self.is_test_column_name)),
                ]
            )
            result_all = self.fit_transform(df_all)
            assert len(result_all) == len(df_all)
            result_train = result_all[: len(df)]
            result_test = result_all[len(df) :]
            train_save_path = self.model_dir / "result_train.arrow"
            result_train.write_ipc(train_save_path)
            self.logger.info(train_save_path)
            test_save_path = self.model_dir / "result_test.arrow"
            result_test.write_ipc(test_save_path)
            self.logger.info(test_save_path)
            return result_train
        else:
            result_test = pl.read_ipc(self.model_dir / "result_test.arrow")
            return result_test

    @property
    def is_test_column_name(self):
        """追加する列名。"""
        return "_is_test_"

    @abc.abstractmethod
    def fit_transform(self, df_all: pl.DataFrame) -> pl.DataFrame:
        """処理"""


class ModelStep(Step, metaclass=abc.ABCMeta):
    """機械学習モデルなど、train/testで別々の処理をするステップ。

    Examples:
        ::

            class Step(pytoolkit.pipelines.ModelStep):
                def __init__(self):
                    super().__init__()
                    self.depends_on = ["xxx"]

                def train(self, df_train: pl.DataFrame) -> pl.DataFrame:
                    return df_train

                def test(self, df_test: pl.DataFrame) -> pl.DataFrame:
                    return df_test

    """

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        if run_type == "train":
            return self.train(df)
        else:
            return self.test(df)

    @abc.abstractmethod
    def train(self, df_train: pl.DataFrame) -> pl.DataFrame:
        """学習してモデルを保存してoofpを返す"""

    @abc.abstractmethod
    def test(self, df_test: pl.DataFrame) -> pl.DataFrame:
        """モデルを読み込んで推論して結果を返す"""


class TrainOnlyStep(Step, metaclass=abc.ABCMeta):
    """グループIDやラベル周りの処理など、"train"のみ実装されているステップ。

    Examples:
        ::

            class Step(pytoolkit.pipelines.TrainOnlyStep):
                def __init__(self):
                    super().__init__()
                    self.depends_on = ["xxx"]

                def train(self, df_train: pl.DataFrame) -> pl.DataFrame:
                    return df_train

    """

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        if run_type == "train":
            return self.train(df)
        else:
            raise NotImplementedError()

    @abc.abstractmethod
    def train(self, df: pl.DataFrame) -> pl.DataFrame:
        """学習してモデルを保存してoofpを返す"""
