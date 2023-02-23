"""All metric configurations used in the project."""

from .config import MetricConfig

SEQEVAL_MICRO_F1 = MetricConfig(
    name="micro_f1",
    pretty_name="Micro-average F1-score",
    huggingface_id="seqeval",
    results_key="overall_f1",
    compute_kwargs=dict(zero_division=1),
    postprocessing_fn=lambda raw_score: f"{100 * raw_score:.2f}%",
)


SEQEVAL_MICRO_F1_NO_MISC = MetricConfig(
    name="micro_f1_no_misc",
    pretty_name="Micro-average F1-score without MISC tags",
    huggingface_id="seqeval",
    results_key="overall_f1",
    compute_kwargs=dict(zero_division=1),
    postprocessing_fn=lambda raw_score: f"{100 * raw_score:.2f}%",
)


MCC = MetricConfig(
    name="mcc",
    pretty_name="Matthew's Correlation Coefficient",
    huggingface_id="matthews_correlation",
    results_key="matthews_correlation",
    postprocessing_fn=lambda raw_score: f"{100 * raw_score:.2f}%",
)


MACRO_F1 = MetricConfig(
    name="macro_f1",
    pretty_name="Macro-average F1-score",
    huggingface_id="f1",
    results_key="f1",
    compute_kwargs=dict(average="macro"),
    postprocessing_fn=lambda raw_score: f"{100 * raw_score:.2f}%",
)


EXACT_MATCH = MetricConfig(
    name="exact_match",
    pretty_name="Exact match",
    huggingface_id="squad_v2",
    results_key="exact",
    postprocessing_fn=lambda raw_score: f"{raw_score:.2f}%",
)


QA_F1 = MetricConfig(
    name="qa_f1",
    pretty_name="F1-score",
    huggingface_id="squad_v2",
    results_key="f1",
    postprocessing_fn=lambda raw_score: f"{raw_score:.2f}%",
)


EMISSIONS = MetricConfig(
    name="carbon_emissions",
    pretty_name="Carbon emissions, in milligrams of CO2 equivalent per sample",
    huggingface_id="",
    results_key="co2",
    postprocessing_fn=lambda raw_score: f"{raw_score:.4f}",
)


POWER = MetricConfig(
    name="energy_consumed",
    pretty_name="Energy consumed, in milliwatt hours per sample",
    huggingface_id="",
    results_key="power",
    postprocessing_fn=lambda raw_score: f"{raw_score:.4f}",
)

WER = MetricConfig(
    name="word_error_rate",
    pretty_name="Word error rate",
    huggingface_id="wer",
    results_key="wer",
    postprocessing_fn=lambda raw_score: f"{raw_score:.4f}",
)
