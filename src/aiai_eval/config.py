"""Configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class MetricConfig:
    """Configuration for a metric.

    Attributes:
        name (str):
            The name of the metric.
        pretty_name (str):
            A longer prettier name for the metric, which allows cases and spaces. Used
            for logging.
        huggingface_id (str):
            The Hugging Face ID of the metric.
        results_key (str):
            The name of the key used to extract the metric scores from the results
            dictionary.
        compute_kwargs (dict, optional):
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
    """

    name: str
    pretty_name: str
    huggingface_id: str
    results_key: str
    compute_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Label:
    """A label in a dataset task.

    Attributes:
        name (str):
            The name of the label.
        synonyms (list of str):
            The synonyms of the label.
    """

    name: str
    synonyms: List[str]


@dataclass
class TaskConfig:
    """Configuration for a task dataset.

    Attributes:
        name (str):
            The name of the task. Must be lower case with no spaces.
        dataset_name (str):
            The name of the task dataset. Must be lower case with no spaces.
        pretty_dataset_name (str):
            A longer prettier name for the dataset, which allows cases and spaces. Used
            for logging.
        huggingface_id (str):
            The Hugging Face ID of the dataset.
        supertask (str):
            The supertask of the task, describing the overall type of task.
        metrics (sequence of MetricConfig objects):
            The metrics used to evaluate the task.
        labels (sequence of Label objects):
            The labels used in the task.
        id2label (list of str):
            The mapping from ID to label.
        label2id (dict of str to int):
            The mapping from label to ID. This includes all label synonyms as well.
        num_labels (int):
            The number of labels in the dataset.
        label_synonyms (list of list of str):
            The synonyms of all the labels, including the main label.
        split_names (dict of str to str or None)
            A dictionary where keys are 'train', 'val', 'test', and the values are
            the corresponding names of the dataset splits, if the split does not exist
            None is used.
    """

    name: str
    dataset_name: str
    pretty_dataset_name: str
    huggingface_id: str
    supertask: str
    metrics: Sequence[MetricConfig]
    labels: Sequence[Label]
    split_names: Dict[str, Optional[str]]

    @property
    def id2label(self) -> List[str]:
        return [label.name for label in self.labels]

    @property
    def label2id(self) -> Dict[str, int]:
        return {
            syn: idx
            for idx, label in enumerate(self.labels)
            for syn in [label.name] + label.synonyms
        }

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def label_synonyms(self) -> List[List[str]]:
        return [[label.name] + label.synonyms for label in self.labels]


@dataclass
class EvaluationConfig:
    """General benchmarking configuration, across datasets and models.

    Attributes:
        raise_error_on_invalid_model (bool):
            Whether to raise an error if a model is invalid.
        cache_dir (str):
            Directory to store cached models and datasets.
        evaluate_train (bool):
            Whether to evaluate on the training set.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        progress_bar (bool):
            Whether to show a progress bar.
        save_results (bool):
            Whether to save the benchmark results to
            'evaluation_results.json'.
        verbose (bool):
            Whether to print verbose output.
        testing (bool, optional):
            Whether a unit test is being run. Defaults to False.
    """

    raise_error_on_invalid_model: bool
    cache_dir: str
    use_auth_token: Union[bool, str]
    progress_bar: bool
    save_results: bool
    verbose: bool
    testing: bool = False


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        model_id (str):
            The ID of the model.
        revision (str):
            The revision of the model.
        framework (str):
            The framework of the model.
    """

    model_id: str
    revision: str
    framework: str
