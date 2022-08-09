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
        self.message = f"The model {model_id} does not exist on the Hugging Face Hub"
        super().__init__(self.message)
