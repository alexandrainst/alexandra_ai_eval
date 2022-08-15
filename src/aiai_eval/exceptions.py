"""Custom exceptions used in the project."""


class InvalidEvaluation(Exception):
    def __init__(
        self, message: str = "This model cannot be evaluated on the given dataset."
    ):
        self.message = message
        super().__init__(self.message)


class ModelDoesNotExistOnHuggingFaceHubException(Exception):
    def __init__(
        self,
        model_id: str,
    ):
        self.model_id = model_id
        self.message = f"The model {model_id} does not exist on the Hugging Face Hub."
        super().__init__(self.message)


class ModelFetchFailed(Exception):
    def __init__(self, model_id: str, error_msg: str, message: str = ""):
        self.model_id = model_id
        self.error_msg = error_msg
        self.message = (
            message
            if message
            else f"Download of {model_id} from the Hugging Face Hub failed, with "
            f"the following error message: {self.error_msg}."
        )
        super().__init__(self.message)


class InvalidFramework(Exception):
    def __init__(self, framework: str):
        self.framework = framework
        self.message = f"The framework {framework} is not supported."
        super().__init__(self.message)


class HuggingFaceHubDown(Exception):
    def __init__(self, message: str = "The Hugging Face Hub is currently down."):
        self.message = message
        super().__init__(self.message)


class NoInternetConnection(Exception):
    def __init__(self, message: str = "There is currently no internet connection."):
        self.message = message
        super().__init__(self.message)
