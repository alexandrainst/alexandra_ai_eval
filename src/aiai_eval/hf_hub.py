"""Functions related to the Hugging Face Hub."""

from typing import Dict, Optional, Sequence, Union


# TODO: port this from ScandEval
def get_model_lists(
    tasks: Optional[Sequence[str]],
    use_auth_token: Union[bool, str],
) -> Dict[str, Sequence[str]]:
    """Fetches up-to-date model lists.
    Args:
        languages (None or sequence of Language objects):
            The language codes of the language to consider. If None then the models
            will not be filtered on language.
        tasks (None or sequence of str):
            The task to consider. If None then the models will not be filtered on task.
        use_auth_token (bool or str):
            The authentication token for the Hugging Face Hub. If a boolean value is
            specified then the token will be fetched from the Hugging Face CLI, where
            the user has logged in through `huggingface-cli login`. If a string is
            specified then it will be used as the token. Defaults to False.
    Returns:
        dict:
            The keys are filterings of the list, which includes all language codes,
            including 'multilingual', all tasks, as well as 'all'. The values are lists
            of model IDs.
    """
    pass
