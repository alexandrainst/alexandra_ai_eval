"""Command-line interface for evaluation of models."""

from typing import Tuple, Union

import click

from .evaluator import Evaluator
from .task_configs import get_all_task_configs


@click.command()
@click.option(
    "--model-id",
    "-m",
    multiple=True,
    help="""The Hugging Face model ID of the model(s) to be benchmarked. The specific
    model version to use can be added after the suffix "@": "<model_id>@v1.0.0". It can
    be a branch name, a tag name, or a commit id (currently only supported for Hugging
    Face models, and it defaults to "main" for latest).""",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    type=click.Choice(list(get_all_task_configs().keys())),
    help="""The name(s) of the task(s) to evaluate.""",
)
@click.option(
    "--auth-token",
    type=str,
    default="",
    show_default=True,
    help="""The authentication token for the Hugging Face Hub. If specified then the
    `--use-auth-token` flag will be set to True.""",
)
@click.option(
    "--use-auth-token",
    is_flag=True,
    show_default=True,
    help="""Whether an authentication token should be used, enabling evaluation of
    private models. Requires that you are logged in via the `huggingface-cli login`
    command.""",
)
@click.option(
    "--track-carbon-emissions",
    "-tce",
    is_flag=True,
    show_default=True,
    help="""Whether to track carbon usage. Remember to set `--country-iso-code` to
    properly calculate carbon emissions""",
)
@click.option(
    "--country-iso-code",
    "-co",
    default="",
    show_default=True,
    help="""The 3-letter alphabet ISO Code of the country where the compute
    infrastructure is hosted. See here:
    https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes""",
)
@click.option(
    "--no-progress-bar",
    "-np",
    is_flag=True,
    show_default=True,
    help="Whether progress bars should be shown.",
)
@click.option(
    "--no-save-results",
    "-ns",
    is_flag=True,
    show_default=True,
    help="Whether results should not be stored to disk.",
)
@click.option(
    "--raise-error-on-invalid-model",
    "-r",
    is_flag=True,
    show_default=True,
    help="Whether to raise an error if a model is invalid.",
)
@click.option(
    "--cache-dir",
    "-c",
    default=".aiai_cache",
    show_default=True,
    help="The directory where models are datasets are cached.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    show_default=True,
    help="Whether extra input should be outputted during benchmarking",
)
def evaluate(
    model_id: Tuple[str],
    task: Tuple[str],
    auth_token: str,
    use_auth_token: bool,
    track_carbon_emissions: bool,
    country_iso_code: str,
    no_progress_bar: bool,
    no_save_results: bool,
    raise_error_on_invalid_model: bool,
    cache_dir: str,
    verbose: bool,
):
    """Benchmark finetuned models."""

    # Raise error if `model_id` or `task` is not specified
    if len(model_id) == 0 or len(task) == 0:
        raise click.UsageError(
            "Please specify at least one model and one task to evaluate."
        )

    # Set up variables
    model_ids = list(model_id)
    tasks = list(task)
    auth: Union[str, bool] = auth_token if auth_token != "" else use_auth_token

    # Initialise the benchmarker class
    evaluator = Evaluator(
        progress_bar=(not no_progress_bar),
        save_results=(not no_save_results),
        raise_error_on_invalid_model=raise_error_on_invalid_model,
        cache_dir=cache_dir,
        use_auth_token=auth,
        verbose=verbose,
        track_carbon_emissions=track_carbon_emissions,
        country_iso_code=country_iso_code,
    )

    # Perform the benchmark evaluation
    evaluator(model_id=model_ids, task=tasks)
