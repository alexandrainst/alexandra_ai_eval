"""Custom exceptions used in the project."""

from typing import Dict


class InvalidEvaluation(Exception):
    def __init__(
        self, message: str = "This model cannot be evaluated on the given dataset."
    ):
        self.message = message
        super().__init__(self.message)


class ModelDoesNotExistOnHuggingFaceHub(Exception):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.message = f"The model {model_id} does not exist on the Hugging Face Hub."
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
    def __init__(self, framework: str):
        self.framework = framework
        self.message = f"The framework {framework} is not supported."
        super().__init__(self.message)


class PreprocessingFailed(Exception):
    def __init__(
        self, message: str = "Preprocessing of the dataset could not be done."
    ):
        self.message = message
        super().__init__(self.message)


class MissingLabel(Exception):
    def __init__(self, label: str, label2id: Dict[str, int]):
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
                f"Received an unsupported model type: {model_type}, "
                "supported types are `nn.Module` and `PretrainedModel`."
            )
        self.model_type = model_type

        super().__init__(self.message)


class MissingCountryISOCode(Exception):
    def __init__(
        self,
        message: str = (
            "The carbon tracker calculates carbon usage based on power consumption, "
            "and the country where the compute infrastructure is hosted. Internet connection "
            "was not available and hence the location of the infrastructure could not be "
            "automatically fetched, because of the location must be set, this is done by setting "
            "the 'country_iso_code' in the config or `--country-iso-code` via the CLI to "
            "the correct ISO code."
        ),
    ):
        self.message = message
        super().__init__(self.message)
