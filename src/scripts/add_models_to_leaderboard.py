"""Script which searches the huggingface_hub for models fitting the supported tasks and add their results to the leaderboard."""

import logging
import os
from csv import writer
from typing import Any, Dict, List, Tuple

import pandas as pd
from huggingface_hub.hf_api import HfApi, ModelFilter

from alexandra_ai_eval.evaluator import Evaluator
from alexandra_ai_eval.task_configs import get_all_task_configs

logger = logging.getLogger(__name__)


def define_searches(task_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Define the searches to be performed on the huggingface_hub.

    Args:
        task_mapping (dict):
            A mapping between names of dataset tasks and their configurations.

        Returns:
            list:
                A list of searches to be performed on the huggingface_hub.
    """
    searches = []
    languages = ["da", "multilingual", "no", "sv", "nn", "nb"]
    for task_name, task_config in task_mapping.items():
        search: Dict[str, List[Dict[str, Any]]] = {task_name: []}
        for language in languages:
            # Get supertask, and check that it is correct.
            if task_config.supertask == "sequence-classification":
                supertask = "text-classification"
            else:
                supertask = task_config.supertask

            # Add search terms, some tasks have no search terms, as they are examples of general supertasks.
            if not task_config.search_terms:
                search[task_name].append(
                    {
                        "search": "",
                        "filter": ModelFilter(task=supertask, language=language),
                    }
                )
            else:
                for search_term in task_config.search_terms:
                    search[task_name].append(
                        {
                            "search": search_term,
                            "filter": ModelFilter(task=supertask, language=language),
                        }
                    )

        searches.append(search)
    return searches


def prepare_cache_and_get_succeeded_and_failed_models(
    cache_dir: str, output_path: str
) -> Tuple[Any, bool, List[str], Any]:
    """
    Prepare cache and get succeeded and failed models.

    Args:
        cache_dir (str): Path to cache directory.
        output_path (str): Path to output directory.

        Returns:
        failed_models_csv_writer (csv.writer): Writer for failed_models.csv.
        failed_models_csv_is_new (bool): True if failed_models.csv is new, False if it already existed.
        models_ids_evaluated (list): List of model ids which have already been evaluated.
        evaluated_models_csv_writer (csv.writer): Writer for evaluated_models.csv.
    """
    # check if output_path exists, if not, create it.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # check if cache_dir exists, if not, create it.
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # make folder in cache_dir for evaluated models list
    if not os.path.exists(f"{cache_dir}/evaluated_models"):
        os.makedirs(f"{cache_dir}/evaluated_models")

    # Check if we already have a list of failed models, if so, we load it.
    failed_models_csv_is_new = False
    try:
        failed_models_csv = open(f"{output_path}/failed_models.csv", "a")
    except FileNotFoundError:
        failed_models_csv = open(f"{output_path}/failed_models.csv", "w+")

    # Check if we already have a list of evaluated_models, if so, we load it.
    try:
        models_ids_evaluated_csv = open(
            f"{cache_dir}/evaluated_models/evaluated_models.csv", "a"
        )
    except FileNotFoundError:
        models_ids_evaluated_csv = open(
            f"{cache_dir}/evaluated_models/evaluated_models.csv", "w+"
        )

    # If failed_models.csv is empty, we add a header.
    if failed_models_csv.tell() == 0:
        failed_models_csv_is_new = True
        failed_models_csv_writer = writer(failed_models_csv)
        failed_models_csv_writer.writerow(["model_id", "error"])

    # If evaluated_models.csv is empty, we add a header.
    if models_ids_evaluated_csv.tell() == 0:
        evaluated_models_csv_writer = writer(models_ids_evaluated_csv)
        evaluated_models_csv_writer.writerow(["model_id"])

    # Load the model_ids we have already evaluated.
    models_ids_evaluated_df = pd.read_csv(
        f"{cache_dir}/evaluated_models/evaluated_models.csv"
    )
    models_ids_evaluated = models_ids_evaluated_df["model_id"].tolist()

    return (
        failed_models_csv_writer,
        failed_models_csv_is_new,
        models_ids_evaluated,
        evaluated_models_csv_writer,
    )


def main(cache_dir: str = ".alexandra_ai_cache", output_path: str = "output"):
    """
    Main function of the script. Searches the huggingface_hub for models fitting the supported tasks and add their results to the leaderboard.

    Args:
        cache_dir (str): Path to cache directory.
        output_path (str): Path to output directory.
    """
    evaluator = Evaluator()
    task_mapping = get_all_task_configs()

    # Get all models from huggingface_hub.
    searches = define_searches(task_mapping)

    # Prepare cache and get succeeded and failed models.
    (
        failed_models_csv_writer,
        failed_models_csv_is_new,
        models_ids_evaluated,
        evaluated_models_csv_writer,
    ) = prepare_cache_and_get_succeeded_and_failed_models(
        cache_dir=cache_dir, output_path=output_path
    )

    # Loop through searches
    hf_api = HfApi()
    for search in searches:
        for task, search_input in search.items():
            for search in search_input:
                # Search for models
                search_term = search["search"]
                search_filter = search["filter"]
                if search_term:
                    models = hf_api.list_models(
                        search=search_term,
                        filter=search_filter,
                        sort="downloads",
                    )
                else:
                    models = hf_api.list_models(
                        filter=search_filter,
                        sort="downloads",
                    )

                # Evaluate models
                for model in models:
                    # If we haven't evaluated the model we evaluate it.
                    if model.modelId not in models_ids_evaluated:
                        try:
                            evaluator.evaluate(
                                model_id=model.modelId,
                                task=task,
                            )
                            evaluated_models_csv_writer.writerow([model.modelId])
                            models_ids_evaluated.append(model.modelId)

                        # If we fail to evaluate the model, we add it to a list of failed models.
                        except Exception as e:
                            logger.info(
                                f"Failed to evaluate model: {model.modelId} with error: {e}"
                            )
                            failed_models_csv_writer.writerow([model.modelId, e])
    failed_models_csv_writer.close()

    # If the csv not created during this run, it might contain old failed model_ids, which might have succeeded in this run.
    # we therefore check if there is any model_ids in the csv which we have been succesfully evaluated in this run, and remove them.
    if not failed_models_csv_is_new:
        failed_models_df = pd.read_csv(f"{output_path}/failed_models.csv")
        failed_models_df = failed_models_df.drop(
            failed_models_df[
                failed_models_df["model_id"].isin(models_ids_evaluated)
            ].index
        )
        failed_models_df.to_csv(f"{output_path}/failed_models.csv", index=False)


if __name__ == "__main__":
    main()
