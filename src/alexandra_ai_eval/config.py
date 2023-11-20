"""Configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Callable

from .enums import CountryCode, Device, Framework, Modality
from .utils import get_available_devices


@dataclass
class LabelConfig:
    """Configuration for a label in a dataset task.

    Attributes:
        name:
            The name of the label.
        synonyms:
            The synonyms of the label.
    """

    name: str
    synonyms: list[str]


@dataclass
class MetricConfig:
    """Configuration for a metric.

    Attributes:
        name:
            The name of the metric.
        pretty_name:
            A longer prettier name for the metric, which allows cases and spaces. Used
            for logging.
        huggingface_id:
            The Hugging Face ID of the metric.
        results_key:
            The name of the key used to extract the metric scores from the results
            dictionary.
        postprocessing_fn:
            A function that is applied to the metric scores after they are extracted
            from the results dictionary. Must take a single float as input and return
            a single string.
        compute_kwargs:
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
    """

    name: str
    pretty_name: str
    huggingface_id: str
    results_key: str | tuple[str, ...]
    postprocessing_fn: Callable[[float], str]
    compute_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a task dataset.

    Attributes:
        name:
            The name of the task. Must be lower case with no spaces.
        pretty_name:
            A longer prettier name for the task, which allows cases and spaces. Used
            for logging.
        huggingface_id:
            The Hugging Face ID of the dataset associated with the task.
        huggingface_subset:
            The subset of the Hugging Face dataset associated with the task. Defaults
            to None.
        supertask:
            The supertask of the task, describing the overall type of task.
        modality:
            The modality of the input data.
        metrics:
            The metrics used to evaluate the task.
        labels:
            The labels used in the task.
        feature_column_names:
            The names of the feature columns for the dataset.
        label_column_name:
            The name of the label column for the dataset.
        test_name:
            The name of the test split of the task. If None, the task has no test
            split.
        id2label:
            The mapping from ID to label.
        label2id:
            The mapping from label to ID. This includes all label synonyms as well.
        num_labels:
            The number of labels in the dataset.
        label_synonyms:
            The synonyms of all the labels, including the main label.
        architectures:
            The architectures that can be used to solve the task. If None then
            it defaults to the list containing only the name of the supertaks. Defaults
            to None.
        search_terms:
            The search terms used to find the task on the Hugging Face Hub. Defaults
            to an empty list.
    """

    name: str
    huggingface_id: str
    huggingface_subset: str | None
    supertask: str
    modality: Modality
    metrics: list[MetricConfig]
    labels: list[LabelConfig]
    feature_column_names: list[str]
    label_column_name: str
    test_name: str | None
    architectures: list[str] | None = None
    search_terms: list[str] = field(default_factory=list)

    @property
    def pretty_name(self) -> str:
        return self.name.replace("-", " ")

    @property
    def id2label(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def label2id(self) -> dict[str, int]:
        return {
            syn: idx
            for idx, label in enumerate(self.labels)
            for syn in [label.name] + label.synonyms
        }

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def label_synonyms(self) -> list[list[str]]:
        return [[label.name] + label.synonyms for label in self.labels]


@dataclass
class EvaluationConfig:
    """General benchmarking configuration, across datasets and models.

    Attributes:
        raise_error_on_invalid_model:
            Whether to raise an error if a model is invalid.
        cache_dir:
            Directory to store cached models and datasets.
        evaluate_train:
            Whether to evaluate on the training set.
        token:
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
        progress_bar:
            Whether to show a progress bar.
        save_results:
            Whether to save the benchmark results to
            'evaluation_results.json'.
        verbose:
            Whether to print verbose output.
        track_carbon_usage:
            Whether to track carbon usage.
        country_code:
            The 3-letter alphabet ISO Code of the country where the compute
            infrastructure is hosted. Only relevant if no internet connection is
            available. Only relevant if `track_carbon_emissions` is set to True. A list
            of all such codes are available here:
            https://en.wikipedia.org/wiki/list_of_ISO_3166_country_codes
        prefer_device:
            The device to prefer when evaluating the model. If the device is not
            available then another device will be used. Can be "cuda", "mps" and "cpu".
            Defaults to "cuda".
        only_return_log:
            Whether to only return the log. Defaults to False.
        architecture_fname:
            The name of the architecture file, if local models are used. If None, the
            architecture file will be automatically detected as the first Python script
            in the model directory. Defaults to None.
        weight_fname:
            The name of the file containing the model weights, if local models are
            used. If None, the weight file will be automatically detected as the first
            "*.bin" file in the model directory. Defaults to None.
        testing:
            Whether a unit test is being run. Defaults to False.
    """

    raise_error_on_invalid_model: bool
    cache_dir: str
    token: bool | str
    progress_bar: bool
    save_results: bool
    verbose: bool
    track_carbon_emissions: bool
    country_code: CountryCode
    prefer_device: Device
    only_return_log: bool = False
    architecture_fname: str | None = None
    weight_fname: str | None = None
    testing: bool = False

    @property
    def device(self) -> str:
        """The compute device to use for the evaluation.

        Returns:
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
        model_id:
            The ID of the model.
        tokenizer_id:
            The ID of the tokenizer.
        processor_id:
            The ID of the processor.
        revision:
            The revision of the model.
        framework:
            The framework of the model.
        id2label:
            The model's mapping from ID to label. If None, the model does not have a
            mapping from ID to label.
        label2id:
            The model's mapping from label to ID. If None, the model does not have a
            mapping from label to ID. Defaults to None.
        num_labels:
            The number of labels in the model. If None, the model does not have a
            mapping between labels and IDs.
    """

    model_id: str
    tokenizer_id: str
    processor_id: str | None
    revision: str
    framework: Framework
    id2label: list[str] | None
    label2id: dict[str, int] | None = None

    @property
    def num_labels(self) -> int | None:
        if self.id2label is None:
            return None
        return len(self.id2label)
