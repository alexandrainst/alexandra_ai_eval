"""Functions related to the Hugging Face Hub."""

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def model_exists_on_hf_hub(model_id: str) -> bool:
    hf_api = HfApi()
    try:
        hf_api.model_info(model_id)
        return True
    except RepositoryNotFoundError:
        return False
