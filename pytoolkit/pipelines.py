"""テーブルデータのパイプライン。(experimental)"""
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


class Step(metaclass=abc.ABCMeta):
    """パイプラインの各ステップ。

    Attributes:
        logger: ロガー

    """

    depends_on: "list[StepType]" = []
    """依存先ステップ"""

    def __init__(self, pipeline: "Pipeline", depend_steps: "list[Step]") -> None:
        self._pipeline = pipeline
        self.depend_steps = depend_steps
        self.logger = logging.getLogger(__name__ + "." + self.name)
        self.fine_refcount = 0

    @property
    def name(self) -> str:
        """ステップ名"""
        return self.__class__.__name__

    def run_name(self, run_type: RunType) -> str:
        """ステップの実行名を作成して返す"""
        fine_suffix = "-fine" if self._pipeline.fine and self.has_fine(run_type) else ""
        return f"{self.name}.{run_type}{fine_suffix}"

    @property
    def use_file_cache(self) -> bool:
        """結果をファイルにキャッシュするのか否か"""
        return True

    @property
    def use_memory_cache(self) -> bool:
        """結果をメモリにキャッシュするのか否か"""
        return True

    def has_fine(self, run_type: RunType) -> bool:
        """fine=Trueな場合に特別な処理をするのか否か"""
        del run_type
        return False

    @property
    def fine(self) -> bool:
        """高精度な学習・推論を行うのか否か"""
        self.fine_refcount += 1
        return self._pipeline.fine

    @property
    def model_dir(self) -> pathlib.Path:
        """モデルを保存したりしたいときのディレクトリ"""
        fine_suffix = "-fine" if self._pipeline.fine and self.has_fine("train") else ""
        return self._pipeline.models_dir / (self.name + fine_suffix)

    @abc.abstractmethod
    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理。Pipeline経由で呼び出される。"""

    def invoke(
        self,
        step_types: "typing.Type[Step] | list[typing.Type[Step]]",
        run_type: RunType | None = None,
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
        return self._pipeline.run(step_types, run_type, cache)

    # TODO: model_dir


# Stepのクラス
StepType = type[Step]


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
        self.memory_cache: dict[tuple[StepType, RunType], pl.DataFrame] = {}
        self.steps: dict[StepType, Step] = {}
        self.run_type_stack: list[RunType] = []
        self.logfmt = (
            "%(asctime)s [%(levelname)-5s] %(message)s"
            " <%(name)s> %(filename)s:%(lineno)d"
        )

    def run_all(
        self,
        steps: StepType | list[StepType] | Step | list[Step],
        cache: typing.Literal["use", "ignore", "disable"] = "use",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """ステップの実行。

        Args:
            steps: 実行するステップのクラス
            run_type: 実行するステップの種類
            cache: ignoreにするとキャッシュがあっても読み込まない、disableにすると保存もしない。(伝播はしない)

        Returns:
            処理結果(train, test)

        """
        steps = self._instantiate(steps)
        df_train = self.run(steps, "train", cache)
        df_test = self.run(steps, "test", cache)
        return df_train, df_test

    def run(
        self,
        steps: StepType | list[StepType] | Step | list[Step],
        run_type: RunType | None,
        cache: typing.Literal["use", "ignore", "disable"] = "use",
    ) -> pl.DataFrame:
        """ステップの実行。

        Args:
            step_type: 実行するステップのクラス (複数指定可)
            run_type: 実行するステップの種類。省略時は現在実行中のステップ。
            cache: ignoreにするとキャッシュがあっても読み込まない、disableにすると保存もしない。(伝播はしない)

        Returns:
            処理結果

        """
        steps = self._instantiate(steps)
        if len(steps) <= 0:
            return pl.DataFrame()
        if run_type is None:
            assert len(self.run_type_stack) > 0, "依存関係の最上位ではrun_typeは省略不可"
            run_type = self.run_type_stack[-1]
        self.run_type_stack.append(run_type)
        try:
            return pl.concat(
                [self._run(step, run_type, cache) for step in steps], how="horizontal"
            )
        finally:
            self.run_type_stack.pop()

    def _instantiate(
        self, steps: StepType | list[StepType] | Step | list[Step]
    ) -> list[Step]:
        """stepsのインスタンス化。"""
        if not isinstance(steps, list):
            steps = [steps]  # type: ignore[assignment]
        assert isinstance(steps, list)
        return [s if isinstance(s, Step) else self._get_step(s) for s in steps]

    def _get_step(self, step_type: StepType) -> Step:
        """Stepのインスタンス化"""
        # インスタンス化済みならそれを返す
        step = self.steps.get(step_type)
        if step is not None:
            return step

        # 依存関係のインスタンス化
        depend_steps = [self._get_step(t) for t in step_type.depends_on]

        # step_typeのインスタンス化
        step = step_type(pipeline=self, depend_steps=depend_steps)
        self.steps[step_type] = step
        return step

    def _run(
        self,
        step: Step,
        run_type: RunType,
        cache: typing.Literal["use", "ignore", "disable"],
    ) -> pl.DataFrame:
        step_run_name = step.run_name(run_type)
        # ファイルキャッシュにあれば読んで返す
        cache_path = self.cache_dir / f"{step_run_name}.arrow"
        if cache == "use" and step.use_file_cache:
            if self._is_cache_valid(step, run_type):
                logger.info(f"'{step_run_name}' load cache: {cache_path}")
                return pl.read_ipc(cache_path)

        # メモリキャッシュにあれば返す
        if step.use_memory_cache:
            result = self.memory_cache.get((step.__class__, run_type))
            if result is not None:
                logger.info(f"'{step_run_name}' get memory cache")
                return result

        # ステップの実行
        result = self._run_step(step, run_type, step_run_name)

        # メモリキャッシュに保存
        if step.use_memory_cache:
            self.memory_cache[(step.__class__, run_type)] = result

        # ファイルキャッシュに保存
        if cache in ("use", "ignore") and step.use_file_cache:
            logger.info(f"'{step_run_name}' save cache: {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.write_ipc(cache_path)
        return result

    def _is_cache_valid(
        self, step: Step, run_type: RunType, base_cache_time: float | None = None
    ) -> bool:
        """キャッシュの有効性チェック"""
        # キャッシュを使わない場合は何であってもOKということにする
        if not step.use_file_cache:
            return True
        # キャッシュが無ければ無効
        step_run_name = step.run_name(run_type)
        cache_path = self.cache_dir / f"{step_run_name}.arrow"
        if not cache_path.exists():
            return False
        # 有効期限の簡易チェック。ソースコードの方が新しければNG。
        # ソースコード上の依存関係とかまでは追えないので注意。
        cache_time = cache_path.stat().st_mtime
        code_time = pathlib.Path(inspect.getfile(step.__class__)).stat().st_mtime
        if cache_time <= code_time:
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
            for s in step.depend_steps
        ):
            return False
        # ここまで来たらOK
        return True

    def _run_step(
        self, step: Step, run_type: RunType, step_run_name: str
    ) -> pl.DataFrame:
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
        logger.info(f"'{step_run_name}' start")
        start_time = time.perf_counter()
        try:
            # 依存関係の実行
            df = self.run(step.depend_steps, run_type)
            # ステップの実行
            step.fine_refcount = 0
            result = step.run(df, run_type)
            if step.has_fine(run_type) == (step.fine_refcount == 0):
                logger.fatal(
                    f"'{step_run_name}' fine refcount error: {run_type=}"
                    f" has_fine={step.has_fine(run_type)} ref={step.fine_refcount=}"
                )
        finally:
            # ログを戻す
            logger.info(
                f"'{step_run_name}' done in {time.perf_counter() - start_time:.0f} s"
            )
            root_logger.removeHandler(file_handler)
        return result


class TransformStep(Step, metaclass=abc.ABCMeta):
    """データ変換ステップ。"""

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        del run_type  # noqa
        return self.transform(df)

    @abc.abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """変換処理"""


class FitTransformStep(Step, metaclass=abc.ABCMeta):
    """特徴量作成ステップ。"""

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        save_path = self.transformer_save_path
        if run_type == "train":
            transformer = self.fit(df)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_transformer(transformer, save_path)
        else:
            transformer = self.load_transformer(save_path)
        return self.transform(transformer, df, run_type)

    @abc.abstractmethod
    def fit(self, df_train: pl.DataFrame) -> typing.Any:
        """訓練＆保存"""

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

    @abc.abstractmethod
    def transform(
        self, transformer, df: pl.DataFrame, run_type: RunType
    ) -> pl.DataFrame:
        """変換処理"""


class AllDataStep(Step, metaclass=abc.ABCMeta):
    """trainで全データを使って処理するステップ。_is_test_というbool列を追加して結合したものがdf_allとして渡される。"""

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        if run_type == "train":
            df_test = self.invoke(self.__class__.depends_on, "test")
            df_all = pl.concat(
                [
                    df.with_columns([pl.lit(False).alias(self.is_test_column_name)]),
                    df_test.with_columns(
                        [pl.lit(True).alias(self.is_test_column_name)]
                    ),
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


class ComposeStep(Step):
    """複数の特徴量をまとめたりするだけのステップ。派生クラスでdepens_onだけ定義する。"""

    @property
    def use_file_cache(self) -> bool:
        """結果をファイルにキャッシュするのか否か"""
        return False

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        del run_type  # noqa
        return df


class ModelStep(Step, metaclass=abc.ABCMeta):
    """機械学習モデルなど、train/testで別々の処理をするステップ。"""

    def run(self, df: pl.DataFrame, run_type: RunType) -> pl.DataFrame:
        """当該ステップの処理"""
        if run_type == "train":
            return self.train(df)
        else:
            return self.test(df)

    @abc.abstractmethod
    def train(self, df_train: pl.DataFrame) -> pl.DataFrame:
        """学習してoofpを返す"""

    @abc.abstractmethod
    def test(self, df_test: pl.DataFrame) -> pl.DataFrame:
        """推論"""