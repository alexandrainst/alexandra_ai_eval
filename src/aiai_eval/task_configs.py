"""All task configurations used in the project."""

from typing import Dict

from .config import Label, TaskConfig
from .metric_configs import MACRO_F1, MCC, SEQEVAL_MICRO_F1, SEQEVAL_MICRO_F1_NO_MISC


def get_all_task_configs() -> Dict[str, TaskConfig]:
    """Get a list of all the dataset tasks.

    Returns:
        dict:
            A mapping between names of dataset tasks and their configurations.
    """
    return {cfg.name: cfg for cfg in globals().values() if isinstance(cfg, TaskConfig)}


NER = TaskConfig(
    name="ner",
    pretty_name="named entity recognition",
    huggingface_id="dane",
    supertask="token-classification",
    metrics=[SEQEVAL_MICRO_F1, SEQEVAL_MICRO_F1_NO_MISC],
    labels=[
        Label(
            name="O",
            synonyms=[],
        ),
        Label(
            name="B-LOC",
            synonyms=[
                "B-LOCATION",
                "B-PLACE",
                "B-GPELOC",
                "B-GPE_LOC",
                "B-GPE/LOC",
                "B-LOCGPE",
                "B-LOC_GPE",
                "B-LOC/GPE",
                "B-LOCORG",
                "B-LOC_ORG",
                "B-LOC/ORG",
                "B-ORGLOC",
                "B-ORG_LOC",
                "B-ORG/LOC",
                "B-LOCPRS",
                "B-LOC_PRS",
                "B-LOC/PRS",
                "B-PRSLOC",
                "B-PRS_LOC",
                "B-PRS/LOC",
            ],
        ),
        Label(
            name="I-LOC",
            synonyms=[
                "I-LOCATION",
                "I-PLACE",
                "I-GPELOC",
                "I-GPE_LOC",
                "I-GPE/LOC",
                "I-LOCGPE",
                "I-LOC_GPE",
                "I-LOC/GPE",
                "I-LOCORG",
                "I-LOC_ORG",
                "I-LOC/ORG",
                "I-ORGLOC",
                "I-ORG_LOC",
                "I-ORG/LOC",
                "I-LOCPRS",
                "I-LOC_PRS",
                "I-LOC/PRS",
                "I-PRSLOC",
                "I-PRS_LOC",
                "I-PRS/LOC",
            ],
        ),
        Label(
            name="B-ORG",
            synonyms=[
                "B-ORGANIZATION",
                "B-ORGANISATION",
                "B-INST",
                "B-GPEORG",
                "B-GPE_ORG",
                "B-GPE/ORG",
                "B-ORGGPE",
                "B-ORG_GPE",
                "B-ORG/GPE",
                "B-ORGPRS",
                "B-ORG_PRS",
                "B-ORG/PRS",
                "B-PRSORG",
                "B-PRS_ORG",
                "B-PRS/ORG",
                "B-OBJORG",
                "B-OBJ_ORG",
                "B-OBJ/ORG",
                "B-ORGOBJ",
                "B-ORG_OBJ",
                "B-ORG/OBJ",
            ],
        ),
        Label(
            name="I-ORG",
            synonyms=[
                "I-ORGANIZATION",
                "I-ORGANISATION",
                "I-INST",
                "I-GPEORG",
                "I-GPE_ORG",
                "I-GPE/ORG",
                "I-ORGGPE",
                "I-ORG_GPE",
                "I-ORG/GPE",
                "I-ORGPRS",
                "I-ORG_PRS",
                "I-ORG/PRS",
                "I-PRSORG",
                "I-PRS_ORG",
                "I-PRS/ORG",
                "I-OBJORG",
                "I-OBJ_ORG",
                "I-OBJ/ORG",
                "I-ORGOBJ",
                "I-ORG_OBJ",
                "I-ORG/OBJ",
            ],
        ),
        Label(
            name="B-PER",
            synonyms=["B-PERSON"],
        ),
        Label(
            name="I-PER",
            synonyms=["I-PERSON"],
        ),
        Label(
            name="B-MISC",
            synonyms=["B-MISCELLANEOUS"],
        ),
        Label(
            name="I-MISC",
            synonyms=["I-MISCELLANEOUS"],
        ),
    ],
    feature_column_name="text",
    train_name="train",
    val_name="validation",
    test_name="test",
)


SENT = TaskConfig(
    name="sent",
    pretty_name="sentiment classification",
    huggingface_id="DDSC/angry-tweets",
    supertask="text-classification",
    metrics=[MCC, MACRO_F1],
    labels=[
        Label(
            name="NEGATIVE",
            synonyms=["NEG", "NEGATIV", "LABEL_0"],
        ),
        Label(
            name="NEUTRAL",
            synonyms=["NEU", "LABEL_1"],
        ),
        Label(
            name="POSITIVE",
            synonyms=["POS", "POSITIV", "LABEL_2"],
        ),
    ],
    feature_column_name="text",
    train_name="train",
    val_name=None,
    test_name="test",
)
