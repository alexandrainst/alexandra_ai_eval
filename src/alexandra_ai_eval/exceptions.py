"""Custom exceptions used in the project."""

from .enums import Framework


class InvalidEvaluation(Exception):
    def __init__(
        self, message: str = "This model cannot be evaluated on the given dataset."
    ):
        self.message = message
        super().__init__(self.message)


class ModelDoesNotExist(Exception):
    def __init__(self, model_id: str, message: str = ""):
        self.model_id = model_id
        self.message = (
            message
            if message
            else (
                f"The model ID '{model_id}' is not a valid model ID on the Hugging "
                "Face Hub, nor is it a valid spaCy model ID. In case of a Hugging "
                "Face model, please check the model ID, and try again. In case of a "
                "spaCy model, please make sure that you have spaCy installed, and "
                "that the model is installed on your system."
            )
        )
        super().__init__(self.message)


class ModelIsPrivate(Exception):
    def __init__(self, model_id: str, message: str = ""):
        self.model_id = model_id
        self.message = (
            message
            if message
            else (
                f"The model ID '{model_id}' is a private model on the Hugging Face "
                "Hub. Please make sure that you have the correct credentials, are "
                "logged in to the Hugging Face Hub via `huggingface-cli login`, and "
                "ensure that `use_auth_token` is set (`--use-auth-token` in the CLI)."
            )
        )
        super().__init__(self.message)


class ModelFetchFailed(Exception):
    def __init__(self, model_id: str, error_msg: str, message: str = ""):
        self.model_id = model_id
        self.error_msg = error_msg
        if message != "":
            self.message = message
        else:
            self.message = (
                f"Download of {model_id} from the Hugging Face Hub failed, with "
                f"the following error message: {self.error_msg}."
            )
        super().__init__(self.message)


class InvalidFramework(Exception):
    def __init__(self, framework: Framework | str):
        self.framework = framework
        self.message = f"The framework {str(framework)} is not supported."
        super().__init__(self.message)


class PreprocessingFailed(Exception):
    def __init__(
        self, message: str = "Preprocessing of the dataset could not be done."
    ):
        self.message = message
        super().__init__(self.message)


class MissingLabel(Exception):
    def __init__(self, label: str, label2id: dict[str, int]):
        self.label = label
        self.label2id = label2id
        self.message = (
            f"One of the labels in the dataset, {self.label}, does "
            f"not occur in the label2id dictionary {self.label2id}."
        )
        super().__init__(self.message)


class HuggingFaceHubDown(Exception):
    def __init__(self, message: str = "The Hugging Face Hub is currently down."):
        self.message = message
        super().__init__(self.message)


class NoInternetConnection(Exception):
    def __init__(self, message: str = "There is currently no internet connection."):
        self.message = message
        super().__init__(self.message)


class UnsupportedModelType(Exception):
    def __init__(self, model_type: str, message: str = ""):
        if message != "":
            self.message = message
        else:
            self.message = (
                f"Received an unsupported model type: {model_type}, supported types "
                "are `nn.Module` and `PretrainedModel`."
            )
        self.model_type = model_type

        super().__init__(self.message)


class MissingCountryISOCode(Exception):
    def __init__(
        self,
        message: str = (
            "The carbon tracker calculates carbon usage based on power consumption, "
            "and the country where the compute infrastructure is hosted. Internet "
            "connection was not available and hence the location of the infrastructure "
            "could not be automatically fetched, because of the location must be set, "
            "this is done by setting the 'country_code' in the config or "
            "`--country-iso-code` via the CLI to the correct ISO code."
        ),
    ):
        self.message = message
        super().__init__(self.message)


class InvalidArchitectureForTask(Exception):
    def __init__(self, architectures: list[str], supertask: str):
        self.architectures = architectures
        self.supertask = supertask
        self.message = (
            "The provided model-id has the following architectures: "
            f"{str(self.architectures)}, none of which fits the provided task's "
            f"supertask: {supertask}. Please provide another model ID which is a "
            f"{supertask}-type model, or provide another task which fits the "
            f"aforementioned architectures."
        )
        super().__init__(self.message)


class WrongFeatureColumnName(Exception):
    def __init__(self, feature_column_names: str | list[str]):
        # Ensure that feature_column_names is a sequence
        if isinstance(feature_column_names, str):
            feature_column_names = [feature_column_names]

        self.feature_column_names = feature_column_names
        self.message = (
            "The provided feature column name(s) "
            f"'{', '.join(self.feature_column_names)}' were incorrect."
        )
        super().__init__(self.message)


class MPSFallbackNotEnabled(Exception):
    def __init__(self):
        self.message = (
            "You are using an MPS backend, such as an M1 GPU, but you have not "
            "enabled the MPS fallback to CPU. Enable this by setting the "
            "PYTORCH_ENABLE_MPS_FALLBACK environment variable to '1'. You can run the "
            "evaluation with this enabled by running `PYTORCH_ENABLE_MPS_FALLBACK=1 "
            "evaluate ...`."
        )
        super().__init__(self.message)


class InvalidTokenizer(Exception):
    def __init__(self, tokenizer_type: str, message: str = ""):
        self.tokenizer_type = tokenizer_type
        self.message = (
            message
            if message
            else (
                f"The provided tokenizer type: {self.tokenizer_type} is not supported."
            )
        )
        super().__init__(self.message)


class InvalidTask(Exception):
    def __init__(self, task: str):
        self.task = task
        self.message = f"The task '{task}' is not supported."
        super().__init__(self.message)


class ModelNotTrainedForTask(Exception):
    def __init__(self, task: str):
        self.task = task
        self.message = f"The model is not trained for the task {self.task}."
        super().__init__(self.message)


class FrameworkCannotHandleTask(Exception):
    def __init__(self, framework: Framework | str, task: str):
        self.task = task
        self.framework = framework
        self.message = (
            f"Evaluation of {str(framework)} models on the {task} task is not "
            "supported."
        )
        super().__init__(self.message)
