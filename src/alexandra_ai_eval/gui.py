"""A simple graphical user interface for evaluating models."""

import click
import gradio as gr

from .evaluator import Evaluator


def evaluate(model_id: str, task: str) -> str:
    """Evaluate a model.

    Args:
        model_id (str):
            The Hugging Face Hub model ID of the model to evaluate.
        task (str):
            The task to evaluate the model on.

    Returns:
        str:
            The evaluation results.

    Raises:
        gradio Error:
            If the evaluation fails.
    """

    # Convert task to the standard task names
    task = task.lower().replace(" ", "-")

    # Create the evaluator
    evaluator = Evaluator(
        progress_bar=True,
        save_results=False,
        raise_error_on_invalid_model=True,
        track_carbon_emissions=True,
        only_return_log=True,
        use_auth_token=True,
    )

    # Evaluate the model on the task
    try:
        results = evaluator.evaluate(model_id=model_id, task=task)[task][model_id]

    except Exception as e:
        if hasattr(e, "message"):
            error_desc = e.message  # type: ignore[attr-defined]
        else:
            error_desc = str(e)
        raise gr.Error(f"{type(e)}: {error_desc}")

    # Return the results
    return results  # type: ignore[return-value]


@click.command()
@click.option(
    "--cache-examples",
    is_flag=True,
    show_default=True,
    help="""Whether the examples should be cached or not.""",
)
def main(cache_examples: bool) -> None:
    """Set up and display the graphical user interface.

    Args:
        cache_examples (bool):
            Whether to cache examples.
    """
    demo = gr.Interface(
        title="AIAI-Eval: Helping You Choose the Right Model",
        description=(
            "This app lets you choose the right model for your task, by evaluating a "
            "given finetuned machine learning model from the Hugging Face Hub on a "
            "specified task."
        ),
        fn=evaluate,
        inputs=[
            gr.Text(
                label="Hugging Face model ID",
                placeholder="Insert Hugging Face model ID",
                max_lines=1,
            ),
            gr.Dropdown(
                label="Task",
                value="Sentiment classification",
                choices=[
                    "Sentiment classification",
                    "Offensive text classification",
                    "Named entity recognition",
                    "Question answering",
                ],
            ),
        ],
        outputs=gr.Text(label="Evaluation results"),
        examples=[
            [
                "pin/senda",
                "Sentiment classification",
            ],
            [
                "DaNLP/da-electra-hatespeech-detection",
                "Offensive text classification",
            ],
            [
                "saattrupdan/nbailab-base-ner-scandi",
                "Named entity recognition",
            ],
            [
                "saattrupdan/xlmr-base-texas-squad-da",
                "Question answering",
            ],
        ],
        cache_examples=cache_examples,
        allow_flagging="auto",
    )
    demo.launch(quiet=True)
