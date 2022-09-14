"""Unit tests for the `hf_hub_utils` module."""

from copy import deepcopy

import pytest
from huggingface_hub import ModelInfo

from src.aiai_eval.config import ModelConfig
from src.aiai_eval.enums import Framework
from src.aiai_eval.exceptions import (
    InvalidEvaluation,
    ModelIsPrivate,
    RepositoryNotFoundError,
)
from src.aiai_eval.hf_hub_utils import (
    get_hf_hub_model_info,
    get_model_config_from_hf_hub,
    load_model_from_hf_hub,
    model_is_private_on_hf_hub,
)


class TestLoadModelFromHfHub:
    def test_raise_error_if_supertask_does_not_correspond_to_automodel_type(
        self, evaluation_config, task_config
    ):
        model_config = ModelConfig(
            model_id="saattrupdan/wav2vec2-xls-r-300m-ftspeech",
            revision="main",
            framework=Framework.PYTORCH,
        )
        task_config_copy = deepcopy(task_config)
        task_config_copy.supertask = "wav-2-vec-2-for-c-t-c"
        with pytest.raises(InvalidEvaluation):
            load_model_from_hf_hub(
                model_config=model_config,
                from_flax=False,
                task_config=task_config_copy,
                evaluation_config=evaluation_config,
            )

    def test_raise_error_if_model_not_accessible(self, evaluation_config, task_config):
        model_config = ModelConfig(
            model_id="invalid-model-id",
            revision="main",
            framework=Framework.PYTORCH,
        )
        with pytest.raises(InvalidEvaluation):
            load_model_from_hf_hub(
                model_config=model_config,
                from_flax=False,
                task_config=task_config,
                evaluation_config=evaluation_config,
            )

    def test_raise_error_if_model_is_private_and_auth_token_not_set(
        self, evaluation_config, task_config
    ):
        # Create copy of the evaluation configuration where the auth token is not set
        evaluation_config_copy = deepcopy(evaluation_config)
        evaluation_config_copy.use_auth_token = False

        model_config = ModelConfig(
            model_id="saattrupdan/nbailab-base-ner-scandi-v2",
            revision="main",
            framework=Framework.PYTORCH,
        )
        with pytest.raises(ModelIsPrivate):
            load_model_from_hf_hub(
                model_config=model_config,
                from_flax=False,
                task_config=task_config,
                evaluation_config=evaluation_config_copy,
            )

    def test_output_dict_has_both_model_and_tokenizer(
        self, evaluation_config, task_config, model_configs
    ):
        for model_config in model_configs:
            if model_config.framework == Framework.PYTORCH:
                model_dict = load_model_from_hf_hub(
                    model_config=model_config,
                    from_flax=False,
                    task_config=task_config,
                    evaluation_config=evaluation_config,
                )
                assert "model" in model_dict
                assert "tokenizer" in model_dict


class TestGetHfHubModelInfo:
    @pytest.fixture(scope="class")
    def existing_model_id(self):
        return "bert-base-uncased"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        return "non-existing-model-id"

    def test_model_info_is_model_info(self, existing_model_id):
        model_info = get_hf_hub_model_info(model_id=existing_model_id)
        assert isinstance(model_info, ModelInfo)

    def test_raise_error_if_model_not_accessible(self, non_existing_model_id):
        with pytest.raises(RepositoryNotFoundError):
            get_hf_hub_model_info(model_id=non_existing_model_id)


class TestModelIsPrivateOnHfHub:
    @pytest.fixture(scope="class")
    def public_model_id(self):
        yield "bert-base-uncased"

    @pytest.fixture(scope="class")
    def private_model_id(self):
        yield "saattrupdan/nbailab-base-ner-scandi-v2"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        yield "invalid-model-id"

    def test_public_model_is_not_private(self, public_model_id):
        assert not model_is_private_on_hf_hub(model_id=public_model_id)

    def test_private_model_is_private(self, private_model_id):
        assert model_is_private_on_hf_hub(model_id=private_model_id)

    def test_non_existing_model_id_does_not_exist(self, non_existing_model_id):
        assert model_is_private_on_hf_hub(model_id=non_existing_model_id) is None


class TestGetModelConfigFromHfHub:
    @pytest.fixture(
        scope="class",
        params=[
            ("bert-base-uncased", "main", Framework.PYTORCH),
            (
                "saattrupdan/wav2vec2-xls-r-300m-ftspeech@"
                "69d0c52dd895e2c1e21518638039638a8539a9cd",
                "69d0c52dd895e2c1e21518638039638a8539a9cd",
                Framework.PYTORCH,
            ),
            (
                "pin/senda",
                "main",
                Framework.PYTORCH,
            ),
        ],
        ids=[
            "normal",
            "with-organization",
            "with-organization-and-revision",
        ],
    )
    def model_id_revision_framework(self, request):
        yield request.param

    @pytest.fixture(scope="class")
    def model_config(self, model_id_revision_framework, evaluation_config):
        model_id, _, _ = model_id_revision_framework
        yield get_model_config_from_hf_hub(
            model_id=model_id, evaluation_config=evaluation_config
        )

    def test_model_config_is_model_config(self, model_config):
        assert isinstance(model_config, ModelConfig)

    def test_model_config_has_correct_information(
        self, model_config, model_id_revision_framework
    ):
        model_id, revision, framework = model_id_revision_framework
        assert model_config.model_id == model_id.split("@")[0]
        assert model_config.revision == revision
        assert model_config.framework == framework
