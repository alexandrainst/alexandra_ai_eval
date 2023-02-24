"""Script which searches the huggingface_hub for models fitting the supported tasks and add their results to the leaderboard."""

import logging
from csv import writer

import pandas as pd
from huggingface_hub.hf_api import HfApi, ModelFilter

from alexandra_ai_eval.evaluator import Evaluator
from alexandra_ai_eval.task_configs import get_all_task_configs

logger = logging.getLogger(__name__)


def main():
    evaluator = Evaluator()
    task_mapping = get_all_task_configs()

    searches = []
    languages = ["da", "multilingual", "no", "sv", "nn", "nb"]
    for task_name, task_config in task_mapping.items():
        search = {task_name: []}
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

    # Check if we already have a list of failed models, if so, we load it.
    csv_is_new = False
    try:
        failed_models_csv = open("failed_models.csv", "a")
    except FileNotFoundError:
        failed_models_csv = open("failed_models.csv", "w+")

    # If failed_models.csv is empty, we add a header.
    if failed_models_csv.tell() == 0:
        csv_is_new = True
        csv_writer = writer(failed_models_csv)
        csv_writer.writerow(["model_id", "error"])

    # Loop through searches
    hf_api = HfApi()
    models_ids_evaluated = []
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
                            models_ids_evaluated.append(model.modelId)

                        # If we fail to evaluate the model, we add it to a list of failed models.
                        except Exception as e:
                            logger.info(
                                f"Failed to evaluate model: {model.modelId} with error: {e}"
                            )
                            csv_writer.writerow([model.modelId, e])
    csv_writer.close()

    # If the csv not created during this run, it might contain old failed model_ids, which might have succeeded in this run.
    # we therefore check if there is any model_ids in the csv which we have been succesfully evaluated in this run, and remove them.
    if not csv_is_new:
        failed_models_df = pd.read_csv("failed_models.csv")
        failed_models_df = failed_models_df.drop(
            failed_models_df[
                failed_models_df["model_id"].isin(models_ids_evaluated)
            ].index
        )
        failed_models_df.to_csv("failed_models.csv", index=False)


if __name__ == "__main__":
    main()
