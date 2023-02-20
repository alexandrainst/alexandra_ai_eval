"""Script which searches the huggingface_hub for models fitting the supported tasks and add their results to the leaderboard."""

from huggingface_hub.hf_api import HfApi, ModelFilter

from src.aiai_eval.evaluator import Evaluator


def main():
    evaluator = Evaluator()

    searches = []

    searches.append(
        {
            "discourse-coherence-classification": [
                {
                    "search": "discourse",
                    "filter": ModelFilter(
                        task="text-classification", language="multilingual"
                    ),
                },
                {
                    "search": "discourse",
                    "filter": ModelFilter(task="text-classification", language="da"),
                },
            ]
        }
    )

    searches.append(
        {
            "sentiment-classification": [
                {
                    "search": "sent",
                    "filter": ModelFilter(
                        task="text-classification", language="multilingual"
                    ),
                },
                {
                    "search": "sent",
                    "filter": ModelFilter(task="text-classification", language="da"),
                },
            ]
        }
    )

    searches.append(
        {
            "question-answering": [
                {
                    "filter": ModelFilter(
                        task="question-answering", language="multilingual"
                    ),
                },
                {
                    "filter": ModelFilter(task="question-answering", language="da"),
                },
            ]
        }
    )

    searches.append(
        {
            "named-entity-recognition": [
                {
                    "search": "ner",
                    "filter": ModelFilter(
                        task="token-classification", language="multilingual"
                    ),
                },
                {
                    "search": "ner",
                    "filter": ModelFilter(task="token-classification", language="da"),
                },
            ]
        }
    )

    searches.append(
        {
            "offensive-text-classification": [
                {
                    "search": "hate",
                    "filter": ModelFilter(
                        task="text-classification", language="multilingual"
                    ),
                },
                {
                    "search": "hate",
                    "filter": ModelFilter(task="text-classification", language="da"),
                },
                {
                    "search": "offensiv",
                    "filter": ModelFilter(
                        task="text-classification", language="multilingual"
                    ),
                },
                {
                    "search": "offensiv",
                    "filter": ModelFilter(task="text-classification", language="da"),
                },
            ]
        }
    )

    searches.append(
        {
            "automatic-speech-recognition": [
                {
                    "filter": ModelFilter(
                        task="automatic-speech-recognition", language="multilingual"
                    ),
                },
                {
                    "filter": ModelFilter(
                        task="automatic-speech-recognition", language="da"
                    ),
                },
            ]
        }
    )

    # Loop through searches
    hf_api = HfApi()
    models_ids_evaluated = []
    for search in searches:
        for task, search_input in search.items():
            for search in search_input:

                # Search for models
                models = hf_api.list_models(
                    search=search_input["search"],
                    filter=search_input["filter"],
                    sort="downloads",
                )

                # Evaluate models
                for model in models:
                    if model.modelId in models_ids_evaluated:
                        continue
                    evaluator.evaluate(
                        model_id=model.modelId,
                        task=task,
                    )
                    models_ids_evaluated.append(model.modelId)


if __name__ == "__main__":
    main()
