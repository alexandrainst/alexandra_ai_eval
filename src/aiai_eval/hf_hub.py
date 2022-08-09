"""Functions related to the Hugging Face Hub."""
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def model_exists_on_hf_hub(model_id: str) -> bool:
    """Function checks if `model_id` exists on Huggingface Hub.

    Args:
        model_id (str): The model ID to check.

    Returns:
        bool: If model exists on Hugginface Hub or not.
    """
    hf_api = HfApi()
    try:
        hf_api.model_info(model_id)
        return True
    except RepositoryNotFoundError:
        return False
