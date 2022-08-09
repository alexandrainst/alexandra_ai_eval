"""Class for named entity recognition tasks."""

from .task import EvaluationDataset


class NEREvaluation(EvaluationDataset):
    """NER dataset.
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
