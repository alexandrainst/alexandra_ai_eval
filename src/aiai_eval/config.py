"""Configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .enums import Device, Framework
from .utils import get_available_devices


@dataclass
class LabelConfig:
    """Configuration for a label in a dataset task.

    Attributes:
        name (str):
            The name of the label.
        synonyms (list of str):
            The synonyms of the label.
    """

    name: str
    synonyms: List[str]


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
        postprocessing_fn (callable):
            A function that is applied to the metric scores after they are extracted
            from the results dictionary. Must take a single float as input and return
            a single string.
        compute_kwargs (dict, optional):
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
    """

    name: str
    pretty_name: str
    huggingface_id: str
    results_key: str
    postprocessing_fn: Callable[[float], str]
    compute_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a task dataset.

    Attributes:
        name (str):
            The name of the task. Must be lower case with no spaces.
        pretty_name (str):
            A longer prettier name for the task, which allows cases and spaces. Used
            for logging.
        huggingface_id (str):
            The Hugging Face ID of the dataset associated with the task.
        huggingface_subset (str or None, optional):
            The subset of the Hugging Face dataset associated with the task. Defaults
            to None.
        supertask (str):
            The supertask of the task, describing the overall type of task.
        metrics (sequence of MetricConfig objects):
            The metrics used to evaluate the task.
        labels (sequence of LabelConfig objects):
            The labels used in the task.
        feature_column_names (list of str):
            The names of the feature columns for the dataset.
        label_column_name (str):
            The name of the label column for the dataset.
        test_name (str or None):
            The name of the test split of the task. If None, the task has no test
            split.
        id2label (list of str):
            The mapping from ID to label.
        label2id (dict of str to int):
            The mapping from label to ID. This includes all label synonyms as well.
        num_labels (int):
            The number of labels in the dataset.
        label_synonyms (list of list of str):
            The synonyms of all the labels, including the main label.
    """

    name: str
    huggingface_id: str
    huggingface_subset: Optional[str]
    supertask: str
    metrics: List[MetricConfig]
    labels: List[LabelConfig]
    feature_column_names: List[str]
    label_column_name: str
    test_name: Optional[str]

    @property
    def pretty_name(self) -> str:
        return self.name.replace("-", " ")

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
        track_carbon_usage (bool):
            Whether to track carbon usage.
        country_iso_code (str):
            The 3-letter alphabet ISO Code of the country where the compute
            infrastructure is hosted. Only relevant if no internet connection is
            available. Only relevant if `track_carbon_emissions` is set to True. A list
            of all such codes are available here:
            https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
        prefer_device (Device):
            The device to prefer when evaluating the model. If the device is not
            available then another device will be used. Can be "cuda", "mps" and "cpu".
            Defaults to "cuda".
        only_return_log (bool, optional):
            Whether to only return the log. Defaults to False.
        testing (bool, optional):
            Whether a unit test is being run. Defaults to False.
    """

    raise_error_on_invalid_model: bool
    cache_dir: str
    use_auth_token: Union[bool, str]
    progress_bar: bool
    save_results: bool
    verbose: bool
    track_carbon_emissions: bool
    country_iso_code: str
    prefer_device: Device
    only_return_log: bool = False
    testing: bool = False

    @property
    def device(self) -> str:
        """The compute device to use for the evaluation.

        Returns:
            str:
                The compute device to use for the evaluation.
        """

        # If CPU is preferred then everything else will be ignored, as the CPU is
        # always available
        if self.prefer_device == Device.CPU:
            return "cpu"

        # Otherwise we fetch a list of available devices
        available_devices = get_available_devices()

        # If MPS is preferred and is available then we use it
        if self.prefer_device == Device.MPS and Device.MPS in available_devices:
            return "mps"

        # Otherwise, we use the best available device, which is the first device
        # present in the list
        return available_devices[0].value


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        model_id (str):
            The ID of the model.
        revision (str):
            The revision of the model.
        framework (Framework):
            The framework of the model.
    """

    model_id: str
    revision: str
    framework: Framework
