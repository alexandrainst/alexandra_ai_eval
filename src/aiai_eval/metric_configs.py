"""All metric configurations used in the project."""

from .config import MetricConfig

SEQEVAL_MICRO_F1 = MetricConfig(
    name="micro_f1",
    pretty_name="Micro-average F1-score",
    huggingface_id="seqeval",
    results_key="overall_f1",
    compute_kwargs=dict(zero_division=1),
)


SEQEVAL_MICRO_F1_NO_MISC = MetricConfig(
    name="micro_f1_no_misc",
    pretty_name="Micro-average F1-score without MISC tags",
    huggingface_id="seqeval",
    results_key="overall_f1",
    compute_kwargs=dict(zero_division=1),
)


MCC = MetricConfig(
    name="mcc",
    pretty_name="Matthew's Correlation Coefficient",
    huggingface_id="matthews_correlation",
    results_key="matthews_correlation",
)


MACRO_F1 = MetricConfig(
    name="macro_f1",
    pretty_name="Macro-average F1-score",
    huggingface_id="f1",
    results_key="f1",
    compute_kwargs=dict(average="macro", zero_division=1),
)


EMISSIONS = MetricConfig(
    name="carbon_emissions",
    pretty_name="Carbon emissions, in milligrams of CO2 equivalent per sample",
    huggingface_id="",
    results_key="co2",
)


POWER = MetricConfig(
    name="energy_consumed",
    pretty_name="Energy consumed, in milliwatt hours per sample",
    huggingface_id="",
    results_key="power",
)
