"""Script which searches the huggingface_hub for models fitting the supported tasks and add their results to the leaderboard."""

import pandas as pd
from huggingface_hub.hf_api import HfApi, ModelFilter

from alexandra_ai_eval.evaluator import Evaluator
from alexandra_ai_eval.task_configs import get_all_task_configs


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

    # Loop through searches
    hf_api = HfApi()
    models_ids_evaluated = []
    failed_models = []
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
                    # Check if we have already evaluated this model
                    if model.modelId in models_ids_evaluated:
                        continue
                    try:
                        evaluator.evaluate(
                            model_id=model.modelId,
                            task=task,
                        )
                        models_ids_evaluated.append(model.modelId)

                    # If we fail to evaluate the model, we add it to a list of failed models.
                    except Exception as e:
                        print(f"Failed to evaluate {model.modelId} because of {e}")
                        failed_models.append(
                            {"model_id": model.modelId, "error": str(e)}
                        )
                        continue

    # Save failed models to csv, for later debugging.
    pd.DataFrame.from_dict(failed_models).to_csv("failed_models.csv", index=False)


if __name__ == "__main__":
    main()
