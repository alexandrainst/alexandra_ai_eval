"""Unit tests for the `hf_hub_utils` module."""

from copy import deepcopy

import pytest
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils import RepositoryNotFoundError

from alexandra_ai_eval.config import ModelConfig
from alexandra_ai_eval.enums import Framework
from alexandra_ai_eval.exceptions import InvalidEvaluation
from alexandra_ai_eval.hf_hub_utils import (
    get_hf_hub_model_info,
    get_model_config_from_hf_hub,
    load_model_from_hf_hub,
    model_exists_on_hf_hub,
)


class TestLoadModelFromHfHub:
    def test_raise_error_if_supertask_does_not_correspond_to_automodel_type(
        self, evaluation_config, task_config
    ):
        model_config = ModelConfig(
            model_id="saattrupdan/wav2vec2-xls-r-300m-ftspeech",
            tokenizer_id="saattrupdan/wav2vec2-xls-r-300m-ftspeech",
            processor_id=None,
            revision="main",
            framework=Framework.PYTORCH,
            id2label=None,
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
            tokenizer_id="invalid-model-id",
            processor_id=None,
            revision="main",
            framework=Framework.PYTORCH,
            id2label=None,
        )
        with pytest.raises(InvalidEvaluation):
            load_model_from_hf_hub(
                model_config=model_config,
                from_flax=False,
                task_config=task_config,
                evaluation_config=evaluation_config,
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

    def test_model_info_is_model_info(self, existing_model_id, evaluation_config):
        model_info = get_hf_hub_model_info(
            model_id=existing_model_id,
            token=evaluation_config.token,
        )
        assert isinstance(model_info, ModelInfo)

    def test_raise_error_if_model_not_accessible(
        self, non_existing_model_id, evaluation_config
    ):
        with pytest.raises(RepositoryNotFoundError):
            get_hf_hub_model_info(
                model_id=non_existing_model_id,
                token=evaluation_config.token,
            )


class TestModelExistsOnHfHub:
    @pytest.fixture(scope="class")
    def existing_model_id(self):
        yield "bert-base-uncased"

    @pytest.fixture(scope="class")
    def non_existing_model_id(self):
        yield "invalid-model-id"

    def test_existing_model_id_exists(self, existing_model_id, evaluation_config):
        assert model_exists_on_hf_hub(
            model_id=existing_model_id,
            token=evaluation_config.token,
        )

    def test_non_existing_model_id_does_not_exist(
        self, non_existing_model_id, evaluation_config
    ):
        assert not model_exists_on_hf_hub(
            model_id=non_existing_model_id,
            token=evaluation_config.token,
        )


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
