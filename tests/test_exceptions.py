"""Unit tests for the `exceptions` module."""

import numpy as np
import pytest

from src.aiai_eval.enums import Framework
from src.aiai_eval.exceptions import (
    FrameworkCannotHandleTask,
    HuggingFaceHubDown,
    InvalidArchitectureForTask,
    InvalidEvaluation,
    InvalidFramework,
    InvalidTask,
    InvalidTokenizer,
    MissingCountryISOCode,
    MissingLabel,
    ModelDoesNotExist,
    ModelFetchFailed,
    ModelIsPrivate,
    ModelNotTrainedForTask,
    MPSFallbackNotEnabled,
    NoInternetConnection,
    PreprocessingFailed,
    UnsupportedModelType,
    WrongFeatureColumnName,
)


class TestInvalidEvaluation:
    """Unit tests for the InvalidEvaluation exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def exception(self):
        yield InvalidEvaluation()

    @pytest.fixture(scope="class")
    def exception_with_message(self, message):
        yield InvalidEvaluation(message=message)

    def test_invalid_evaluation_is_an_exception(self, exception):
        with pytest.raises(InvalidEvaluation):
            raise exception

    def test_default_message(self, exception):
        assert exception.message == (
            "This model cannot be evaluated on the given dataset."
        )

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message


class TestModelDoesNotExist:
    """Unit tests for the ModelDoesNotExist exception class."""

    @pytest.fixture(scope="class")
    def model_id(self):
        yield "test_model_id"

    @pytest.fixture(scope="class")
    def message(self):
        yield "test_message"

    @pytest.fixture(scope="class")
    def exception(self, model_id):
        yield ModelDoesNotExist(model_id=model_id)

    @pytest.fixture(scope="class")
    def exception_with_message(self, model_id, message):
        yield ModelDoesNotExist(model_id=model_id, message=message)

    def test_model_does_not_exist_is_an_exception(self, exception):
        with pytest.raises(ModelDoesNotExist):
            raise exception

    def test_model_id_is_stored(self, exception_with_message, model_id):
        assert exception_with_message.model_id == model_id

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception, model_id):
        message = (
            f"The model ID '{model_id}' is not a valid model ID on the Hugging "
            "Face Hub, nor is it a valid spaCy model ID. In case of a Hugging "
            "Face model, please check the model ID, and try again. In case of a "
            "spaCy model, please make sure that you have spaCy installed, and "
            "that the model is installed on your system."
        )
        assert exception.message == message


class TestModelIsPrivate:
    """Unit tests for the ModelIsPrivate exception class."""

    @pytest.fixture(scope="class")
    def model_id(self):
        yield "test_model_id"

    @pytest.fixture(scope="class")
    def message(self):
        yield "test_message"

    @pytest.fixture(scope="class")
    def exception(self, model_id):
        yield ModelIsPrivate(model_id=model_id)

    @pytest.fixture(scope="class")
    def exception_with_message(self, model_id, message):
        yield ModelIsPrivate(model_id=model_id, message=message)

    def test_model_is_private_is_an_exception(self, exception):
        with pytest.raises(ModelIsPrivate):
            raise exception

    def test_model_id_is_stored(self, exception_with_message, model_id):
        assert exception_with_message.model_id == model_id

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception, model_id):
        message = (
            f"The model ID '{model_id}' is a private model on the Hugging Face "
            "Hub. Please make sure that you have the correct credentials, are "
            "logged in to the Hugging Face Hub via `huggingface-cli login`, and "
            "ensure that `use_auth_token` is set (`--use-auth-token` in the CLI)."
        )
        assert exception.message == message


class TestModelFetchFailed:
    """Unit tests for the ModelFetchFailed exception class."""

    @pytest.fixture(scope="class")
    def model_id(self):
        yield "test_model_id"

    @pytest.fixture(scope="class")
    def error_msg(self):
        yield "test_error_msg"

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message"

    @pytest.fixture(scope="class")
    def exception(self, model_id, error_msg):
        yield ModelFetchFailed(model_id=model_id, error_msg=error_msg)

    @pytest.fixture(scope="class")
    def exception_with_message(self, model_id, error_msg, message):
        yield ModelFetchFailed(model_id=model_id, error_msg=error_msg, message=message)

    def test_model_fetch_failed_is_an_exception(self, exception_with_message):
        with pytest.raises(ModelFetchFailed):
            raise exception_with_message

    def test_model_id_is_stored(self, exception_with_message, model_id):
        assert exception_with_message.model_id == model_id

    def test_error_msg_is_stored(self, exception_with_message, error_msg):
        assert exception_with_message.error_msg == error_msg

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception, model_id, error_msg):
        message = (
            f"Download of {model_id} from the Hugging Face Hub failed, with "
            f"the following error message: {error_msg}."
        )
        assert exception.message == message


class TestInvalidFramework:
    """Unit tests for the InvalidFramework exception class."""

    @pytest.fixture(scope="class", params=["test_framework", Framework.PYTORCH])
    def framework(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def exception(self, framework):
        yield InvalidFramework(framework=framework)

    def test_invalid_framework_is_an_exception(self, exception):
        with pytest.raises(InvalidFramework):
            raise exception

    def test_framework_is_stored(self, exception, framework):
        assert exception.framework == framework

    def test_message_is_stored(self, exception, framework):
        assert exception.message == f"The framework {framework} is not supported."


class TestPreprocessingFailed:
    """Unit tests for the PreprocessingFailed exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def exception(self):
        yield PreprocessingFailed()

    @pytest.fixture(scope="class")
    def exception_with_message(self, message):
        yield PreprocessingFailed(message=message)

    def test_preprocessing_failed_is_an_exception(self, exception):
        with pytest.raises(PreprocessingFailed):
            raise exception

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception):
        assert exception.message == "Preprocessing of the dataset could not be done."


class TestMissingLabel:
    """Unit tests for the MissingLabel exception class."""

    @pytest.fixture(scope="class")
    def label(self):
        yield "TestLabel"

    @pytest.fixture(scope="class")
    def label2id(self):
        yield dict(TestLabel=0)

    @pytest.fixture(scope="class")
    def exception(self, label, label2id):
        yield MissingLabel(label=label, label2id=label2id)

    def test_missing_label_is_an_exception(self, exception):
        with pytest.raises(MissingLabel):
            raise exception

    def test_label_is_stored(self, exception, label):
        assert exception.label == label

    def test_label2id_is_stored(self, exception, label2id):
        assert exception.label2id == label2id

    def test_message_is_stored(self, exception, label, label2id):
        message = (
            f"One of the labels in the dataset, {label}, does not occur in the "
            f"label2id dictionary {label2id}."
        )
        assert exception.message == message


class TestHuggingFaceHubDown:
    """Unit tests for the HuggingFaceHubDown exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def exception(self):
        yield HuggingFaceHubDown()

    @pytest.fixture(scope="class")
    def exception_with_message(self, message):
        yield HuggingFaceHubDown(message=message)

    def test_hugging_face_hub_down_is_an_exception(self, exception):
        with pytest.raises(HuggingFaceHubDown):
            raise exception

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception):
        assert exception.message == "The Hugging Face Hub is currently down."


class TestNoInternetConnection:
    """Unit tests for the NoInternetConnection exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def exception(self):
        yield NoInternetConnection()

    @pytest.fixture(scope="class")
    def exception_with_message(self, message):
        yield NoInternetConnection(message=message)

    def test_no_internet_connection_is_an_exception(self, exception):
        with pytest.raises(NoInternetConnection):
            raise exception

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception):
        assert exception.message == "There is currently no internet connection."


class TestUnsupportedModelType:
    """Unit tests for the UnsupportedModelType exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def model_type(self):
        yield "test_model_type"

    @pytest.fixture(scope="class")
    def exception(self, model_type):
        yield UnsupportedModelType(model_type=model_type)

    @pytest.fixture(scope="class")
    def exception_with_message(self, message, model_type):
        yield UnsupportedModelType(message=message, model_type=model_type)

    def test_unsupported_model_type_is_an_exception(self, exception):
        with pytest.raises(UnsupportedModelType):
            raise exception

    def test_model_type_is_stored(self, exception, model_type):
        assert exception.model_type == model_type

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception, model_type):
        message = (
            f"Received an unsupported model type: {model_type}, supported types are "
            "`nn.Module` and `PretrainedModel`."
        )
        assert exception.message == message


class TestMissingCountryISOCode:
    """Unit tests for the MissingCountryISOCode exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def exception(self):
        yield MissingCountryISOCode()

    @pytest.fixture(scope="class")
    def exception_with_message(self, message):
        yield MissingCountryISOCode(message=message)

    def test_missing_country_iso_code_is_an_exception(self, exception):
        with pytest.raises(MissingCountryISOCode):
            raise exception

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception):
        message = (
            "The carbon tracker calculates carbon usage based on power consumption, "
            "and the country where the compute infrastructure is hosted. Internet "
            "connection was not available and hence the location of the infrastructure "
            "could not be automatically fetched, because of the location must be set, "
            "this is done by setting the 'country_code' in the config or "
            "`--country-iso-code` via the CLI to the correct ISO code."
        )
        assert exception.message == message


class TestInvalidArchitectureForTask:
    """Unit tests for the InvalidArchitectureForTask exception class."""

    @pytest.fixture(
        scope="class",
        params=[
            ["test_model_type"],
            ["test_model_type", "test_model_type2"],
            np.array(["test_model_type"]),
            np.array(["test_model_type", "test_model_type2"]),
        ],
    )
    def architectures(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def supertask(self):
        yield "supertask"

    @pytest.fixture(scope="class")
    def exception(self, architectures, supertask):
        yield InvalidArchitectureForTask(
            architectures=architectures, supertask=supertask
        )

    def test_invalid_architecture_for_task_is_an_exception(self, exception):
        with pytest.raises(InvalidArchitectureForTask):
            raise exception

    def test_supertask_is_stored(self, exception, supertask):
        assert exception.supertask == supertask

    def test_architecture_is_stored(self, exception, architectures):
        if isinstance(architectures, np.ndarray):
            np.testing.assert_equal(exception.architectures, architectures)
        else:
            assert exception.architectures == architectures

    def test_message_is_stored(self, exception, architectures, supertask):
        message = (
            "The provided model-id has the following architectures: "
            f"{str(architectures)}, none of which fits the provided task's "
            f"supertask: {supertask}. Please provide another model ID which is a "
            f"{supertask}-type model, or provide another task which fits the "
            f"aforementioned architectures."
        )
        assert exception.message == message


class TestWrongFeatureColumnName:
    """Unit tests for the WrongFeatureColumnName exception class."""

    @pytest.fixture(
        scope="class",
        params=[
            "feature_name",
            ["feature_name"],
            ["feature_name", "feature_name2"],
            np.array(["feature_name"]),
            np.array(["feature_name", "feature_name2"]),
        ],
    )
    def feature_column_names(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def exception(self, feature_column_names):
        yield WrongFeatureColumnName(feature_column_names=feature_column_names)

    def test_wrong_feature_column_name_is_an_exception(self, exception):
        with pytest.raises(WrongFeatureColumnName):
            raise exception

    def test_model_type_is_stored(self, exception, feature_column_names):
        if isinstance(feature_column_names, str):
            feature_column_names = [feature_column_names]
        if isinstance(feature_column_names, np.ndarray):
            np.testing.assert_equal(
                exception.feature_column_names, feature_column_names
            )
        else:
            assert exception.feature_column_names == feature_column_names

    def test_message_is_stored(self, exception, feature_column_names):
        if isinstance(feature_column_names, str):
            feature_column_names = [feature_column_names]
        message = (
            f"The provided feature column name(s) '{', '.join(feature_column_names)}' "
            "were incorrect."
        )
        assert exception.message == message


class TestMPSFallbackNotEnabled:
    """Unit tests for the MPSFallbackNotEnabled exception class."""

    @pytest.fixture(scope="class")
    def exception(self):
        yield MPSFallbackNotEnabled()

    def test_mps_fallback_not_enabled_is_an_exception(self, exception):
        with pytest.raises(MPSFallbackNotEnabled):
            raise exception

    def test_message_is_stored(self, exception):
        message = (
            "You are using an MPS backend, such as an M1 GPU, but you have not "
            "enabled the MPS fallback to CPU. Enable this by setting the "
            "PYTORCH_ENABLE_MPS_FALLBACK environment variable to '1'. You can run the "
            "evaluation with this enabled by running `PYTORCH_ENABLE_MPS_FALLBACK=1 "
            "evaluate ...`."
        )
        assert exception.message == message


class TestInvalidTokenizer:
    """Unit tests for the InvalidTokenizer exception class."""

    @pytest.fixture(scope="class")
    def message(self):
        yield "Test message."

    @pytest.fixture(scope="class")
    def tokenizer_type(self):
        yield "test_tokenizer_type"

    @pytest.fixture(scope="class")
    def exception(self, tokenizer_type):
        yield InvalidTokenizer(tokenizer_type=tokenizer_type)

    @pytest.fixture(scope="class")
    def exception_with_message(self, message, tokenizer_type):
        yield InvalidTokenizer(message=message, tokenizer_type=tokenizer_type)

    def test_invalid_tokenizer_is_an_exception(self, exception):
        with pytest.raises(InvalidTokenizer):
            raise exception

    def test_tokenizer_is_stored(self, exception, tokenizer_type):
        assert exception.tokenizer_type == tokenizer_type

    def test_custom_message_is_stored(self, exception_with_message, message):
        assert exception_with_message.message == message

    def test_default_message(self, exception, tokenizer_type):
        message = f"The provided tokenizer type: {tokenizer_type} is not supported."
        assert exception.message == message


class TestInvalidTask:
    """Unit tests for the InvalidTask exception class."""

    @pytest.fixture(scope="class")
    def task(self):
        yield "test_task"

    @pytest.fixture(scope="class")
    def exception(self, task):
        yield InvalidTask(task=task)

    def test_invalid_task_is_an_exception(self, exception):
        with pytest.raises(InvalidTask):
            raise exception

    def test_task_is_stored(self, exception, task):
        assert exception.task == task

    def test_message_is_stored(self, exception, task):
        assert exception.message == f"The task '{task}' is not supported."


class TestModelNotTrainedForTask:
    """Unit tests for the ModelNotTrainedForTask exception class."""

    @pytest.fixture(scope="class")
    def task(self):
        yield "test_task"

    @pytest.fixture(scope="class")
    def exception(self, task):
        yield ModelNotTrainedForTask(task=task)

    def test_model_not_trained_for_task_is_an_exception(self, exception):
        with pytest.raises(ModelNotTrainedForTask):
            raise exception

    def test_task_is_stored(self, exception, task):
        assert exception.task == task

    def test_message_is_stored(self, exception, task):
        assert exception.message == f"The model is not trained for the task {task}."


class TestFrameworkCannotHandleTask:
    """Unit tests for the FrameworkCannotHandleTask exception class."""

    @pytest.fixture(scope="class")
    def task(self):
        yield "test_task"

    @pytest.fixture(scope="class", params=["test_framework", Framework.PYTORCH])
    def framework(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def exception(self, task, framework):
        yield FrameworkCannotHandleTask(task=task, framework=framework)

    def test_framework_cannot_handle_task_is_an_exception(self, exception):
        with pytest.raises(FrameworkCannotHandleTask):
            raise exception

    def test_task_is_stored(self, exception, task):
        assert exception.task == task

    def test_framework_is_stored(self, exception, framework):
        assert exception.framework == framework

    def test_message_is_stored(self, exception, task, framework):
        message = (
            f"Evaluation of {str(framework)} models on the {task} task is not "
            "supported."
        )
        assert exception.message == message
