"""Class for named entity recognition tasks."""

from .task import DatasetTask


class TokenClassificationEvaluation(DatasetTask):
    """Token classification benchmark dataset.
    Args:
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
    Attributes:
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.
    """


class NEREvaluation(TokenClassificationEvaluation):
    """Token classification benchmark dataset.
    Args:
        dataset_config (DatasetConfig):
            The dataset configuration.
        benchmark_config (BenchmarkConfig):
            The benchmark configuration.
    Attributes:
        dataset_config (DatasetConfig):
            The configuration of the dataset.
        benchmark_config (BenchmarkConfig):
            The configuration of the benchmark.
    """
