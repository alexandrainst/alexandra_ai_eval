"""All task configurations used in the project."""

from .config import LabelConfig, TaskConfig
from .enums import Modality
from .metric_configs import (
    EXACT_MATCH,
    MACRO_F1,
    MCC,
    QA_F1,
    SEQEVAL_LOC_F1,
    SEQEVAL_MICRO_F1,
    SEQEVAL_MICRO_F1_NO_MISC,
    SEQEVAL_MISC_F1,
    SEQEVAL_ORG_F1,
    SEQEVAL_PER_F1,
    WER,
)


def get_all_task_configs() -> dict[str, TaskConfig]:
    """Get a list of all the dataset tasks.

    Returns:
        A mapping between names of dataset tasks and their configurations.
    """
    return {cfg.name: cfg for cfg in globals().values() if isinstance(cfg, TaskConfig)}


SENT_TRINARY = TaskConfig(
    name="sentiment-trinary-classification",
    huggingface_id="DDSC/angry-tweets",
    huggingface_subset=None,
    supertask="sequence-classification",
    modality=Modality("text"),
    metrics=[MCC, MACRO_F1],
    labels=[
        LabelConfig(
            name="NEGATIVE",
            synonyms=["NEG", "NEGATIV", "LABEL_0"],
        ),
        LabelConfig(
            name="NEUTRAL",
            synonyms=["NEU", "LABEL_1"],
        ),
        LabelConfig(
            name="POSITIVE",
            synonyms=["POS", "POSITIV", "LABEL_2"],
        ),
    ],
    feature_column_names=["text"],
    label_column_name="label",
    test_name="test",
    search_terms=["sent"],
)


SENT_BINARY = TaskConfig(
    name="sentiment-binary-classification",
    huggingface_id="DDSC/angry-tweets-binary",
    huggingface_subset=None,
    supertask="sequence-classification",
    modality=Modality("text"),
    metrics=[MCC, MACRO_F1],
    labels=[
        LabelConfig(
            name="NEGATIVE",
            synonyms=["NEG", "NEGATIV", "LABEL_0"],
        ),
        LabelConfig(
            name="POSITIVE",
            synonyms=["POS", "POSITIV", "LABEL_1"],
        ),
    ],
    feature_column_names=["text"],
    label_column_name="label",
    test_name="test",
    search_terms=["sent"],
)


NER = TaskConfig(
    name="named-entity-recognition",
    huggingface_id="norne",  # TEMP
    huggingface_subset="combined",
    supertask="token-classification",
    modality=Modality("text"),
    metrics=[
        SEQEVAL_MICRO_F1,
        SEQEVAL_MICRO_F1_NO_MISC,
        SEQEVAL_PER_F1,
        SEQEVAL_LOC_F1,
        SEQEVAL_ORG_F1,
        SEQEVAL_MISC_F1,
    ],
    labels=[
        LabelConfig(
            name="O",
            synonyms=["0"],
        ),
        LabelConfig(
            name="B-PER",
            synonyms=["B-PERSON", "1"],
        ),
        LabelConfig(
            name="I-PER",
            synonyms=["I-PERSON", "2"],
        ),
        LabelConfig(
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
                "3",
                "11",
            ],
        ),
        LabelConfig(
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
                "4",
                "12",
            ],
        ),
        LabelConfig(
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
                "5",
                "9",
            ],
        ),
        LabelConfig(
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
                "6",
                "10",
            ],
        ),
        LabelConfig(
            name="B-MISC",
            synonyms=["B-MISCELLANEOUS", "17", "7", "13", "15"],
        ),
        LabelConfig(
            name="I-MISC",
            synonyms=["I-MISCELLANEOUS", "18", "8", "14", "16"],
        ),
    ],
    feature_column_names=["text"],
    label_column_name="ner_tags",
    test_name="test",
    search_terms=["ner"],
)


QA = TaskConfig(
    name="question-answering",
    huggingface_id="alexandrainst/scandiqa",
    huggingface_subset="da",
    supertask="question-answering",
    modality=Modality("text"),
    metrics=[EXACT_MATCH, QA_F1],
    labels=[
        LabelConfig(
            name="START_POSITIONS",
            synonyms=["LABEL_0"],
        ),
        LabelConfig(
            name="END_POSITIONS",
            synonyms=["LABEL_1"],
        ),
    ],
    feature_column_names=["question", "context"],
    label_column_name="answers",
    test_name="test",
    search_terms=[],
)


OFFENSIVE = TaskConfig(
    name="offensive-text-classification",
    huggingface_id="DDSC/dkhate",
    huggingface_subset=None,
    supertask="sequence-classification",
    modality=Modality("text"),
    metrics=[MCC, MACRO_F1],
    labels=[
        LabelConfig(
            name="NOT_OFFENSIVE",
            synonyms=["NOT", "NOT OFFENSIVE", "LABEL_0"],
        ),
        LabelConfig(
            name="OFFENSIVE",
            synonyms=["OFF", "LABEL_1"],
        ),
    ],
    feature_column_names=["text"],
    label_column_name="label",
    test_name="test",
    search_terms=["offensiv", "hate"],
)

DISCOURSE = TaskConfig(
    name="discourse-coherence-classification",
    huggingface_id="alexandrainst/ddisco",
    huggingface_subset=None,
    supertask="sequence-classification",
    modality=Modality("text"),
    metrics=[MCC, MACRO_F1],
    labels=[
        LabelConfig(
            name="LOW_COHERENCE",
            synonyms=["LABEL_0"],
        ),
        LabelConfig(
            name="MEDIUM_COHERENCE",
            synonyms=["LABEL_1"],
        ),
        LabelConfig(
            name="HIGH_COHERENCE",
            synonyms=["LABEL_2"],
        ),
    ],
    feature_column_names=["text"],
    label_column_name="rating",
    test_name="test",
    search_terms=["discourse"],
)

ASR = TaskConfig(
    name="automatic-speech-recognition",
    huggingface_id="mozilla-foundation/common_voice_11_0",
    huggingface_subset="da",
    supertask="automatic-speech-recognition",
    architectures=[
        "wav2-vec2-for-c-t-c",
        "whisper-for-conditional-generation",
    ],
    modality=Modality("audio"),
    metrics=[WER],
    labels=[
        LabelConfig(
            name="LABEL_0",
            synonyms=[],
        ),
        LabelConfig(
            name="LABEL_1",
            synonyms=[],
        ),
    ],
    feature_column_names=["audio"],
    label_column_name="sentence",
    test_name="test",
    search_terms=[],
)
