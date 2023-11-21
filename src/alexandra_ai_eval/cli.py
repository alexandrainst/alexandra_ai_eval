"""Command-line interface for evaluation of models."""

import click

from .country_codes import ALL_COUNTRY_CODES
from .enums import CountryCode, Device
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
    `--token` flag will be set to True.""",
)
@click.option(
    "--token",
    is_flag=True,
    show_default=True,
    help="""Whether an authentication token should be used, enabling evaluation of
    private models. Requires that you are logged in via the `huggingface-cli login`
    command.""",
)
@click.option(
    "--track-carbon-emissions",
    "-co2",
    is_flag=True,
    show_default=True,
    help="""Whether to track carbon usage.""",
)
@click.option(
    "--country-code",
    type=click.Choice([""] + ALL_COUNTRY_CODES),
    default="",
    show_default=True,
    metavar="COUNTRY CODE",
    help="""The 3-letter alphabet ISO Code of the country where the compute
    infrastructure is hosted. Only relevant if no internet connection is available.
    Only relevant if `--track-carbon-emissions` is set. A list of all such codes are
    available here: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes""",
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
    is_flag=True,
    show_default=True,
    help="Whether results should not be stored to disk.",
)
@click.option(
    "--no-send-results-to-leaderboard",
    is_flag=True,
    show_default=True,
    help="Whether results should not be sent to the leaderboard.",
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
    default=".alexandra_ai_cache",
    show_default=True,
    help="The directory where models are datasets are cached.",
)
@click.option(
    "--prefer-device",
    type=click.Choice([device.lower() for device in Device.__members__.keys()]),
    default="cuda",
    show_default=True,
    help="""The device to prefer when evaluating the model. If the device is not
    available then another device will be used.""",
)
@click.option(
    "--architecture-fname",
    type=str,
    default="None",
    help="""The name of the architecture file, if local models are used. If None, the
    architecture file will be automatically detected as the first Python script in the
    model directory. Defaults to None.""",
)
@click.option(
    "--weight-fname",
    type=str,
    default="None",
    help="""The name of the file containing the model weights, if local models are
    used. If None, the architecture file will be automatically detected as the first
    Python script in the model directory. Defaults to None.""",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    show_default=True,
    help="Whether extra input should be outputted during benchmarking",
)
def evaluate(
    model_id: tuple[str],
    task: tuple[str],
    auth_token: str,
    token: bool,
    track_carbon_emissions: bool,
    country_code: str,
    no_progress_bar: bool,
    no_save_results: bool,
    no_send_results_to_leaderboard: bool,
    raise_error_on_invalid_model: bool,
    cache_dir: str,
    prefer_device: str,
    architecture_fname: str,
    weight_fname: str,
    verbose: bool,
):
    """Benchmark finetuned models."""

    # Raise error if `model_id` or `task` is not specified
    if len(model_id) == 0 or len(task) == 0:
        raise click.UsageError(
            "Please specify at least one model and one task to evaluate."
        )

    model_ids = list(model_id)
    tasks = list(task)
    auth: str | bool = auth_token if auth_token != "" else token
    architecture_fname_or_none: str | None = architecture_fname
    weight_fname_or_none: str | None = weight_fname

    if architecture_fname_or_none == "None":
        architecture_fname_or_none = None
    if weight_fname_or_none == "None":
        weight_fname_or_none = None

    evaluator = Evaluator(
        progress_bar=(not no_progress_bar),
        save_results=(not no_save_results),
        send_results_to_leaderboard=(not no_send_results_to_leaderboard),
        raise_error_on_invalid_model=raise_error_on_invalid_model,
        cache_dir=cache_dir,
        token=auth,
        track_carbon_emissions=track_carbon_emissions,
        country_code=CountryCode(country_code.lower()),
        prefer_device=Device(prefer_device.lower()),
        architecture_fname=architecture_fname_or_none,
        weight_fname=weight_fname_or_none,
        verbose=verbose,
    )

    # Perform the evaluation
    evaluator(model_id=model_ids, task=tasks)
