<div align='center'>
 <img src="https://raw.githubusercontent.com/alexandrainst/AlexandraAI/main/gfx/alexandra-ai-logo-dark.svg">
</div>

### Evaluation of Finetuned Models

______________________________________________________________________
[![PyPI Status](https://badge.fury.io/py/alexandra_ai_eval.svg)](https://pypi.org/project/alexandra_ai_eval/)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/alexandra_ai_eval/alexandra_ai_eval.html)
[![License](https://img.shields.io/github/license/alexandrainst/alexandra_ai_eval)](https://github.com/alexandrainst/alexandra_ai_eval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/alexandra_ai_eval)](https://github.com/alexandrainst/alexandra_ai_eval/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-74%25-yellow.svg)](https://github.com/alexandrainst/alexandra_ai_eval/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/alexandra_ai_eval/blob/main/CODE_OF_CONDUCT.md)


## Quickstart

To install the package simply write the following command in your favorite terminal:

```
pip install alexandra-ai-eval
```

### Benchmarking from the Command Line

The easiest way to benchmark pretrained models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:

```
evaluate --model-id <model_id> --task <task>
```

Here `model_id` is the HuggingFace model ID, which can be found on the [HuggingFace
Hub](https://huggingface.co/models), and `task` is the task you want to benchmark the
model on, such as "ner" for named entity recognition. See all options by typing

```
evaluate --help
```

The specific model version to use can also be added after the suffix '@':

```
evaluate --model_id <model_id>@<commit>
```

It can be a branch name, a tag name, or a commit id. It defaults to 'main' for latest.

Multiple models and tasks can be specified by just attaching multiple arguments. Here
is an example with two models:

```
evaluate --model_id <model_id1> --model_id <model_id2> --task ner
```

See all the arguments and options available for the `evaluate` command by typing

```
evaluate --help
```

### Benchmarking from a Script

In a script, the syntax is similar to the command line interface. You simply initialise
an object of the `Evaluator` class, and call this evaluate object with your favorite
models and/or datasets:

```
>>> from alexandra_ai_eval import Evaluator
>>> evaluator = Evaluator()
>>> evaluator('<model_id>', '<task>')
```

## Contributors

If you feel like this package is missing a crucial feature, if you encounter a bug or
if you just want to correct a typo in this readme file, then we urge you to join the
community! Have a look at the [CONTRIBUTING.md](./CONTRIBUTING.md) file, where you can
check out all the ways you can contribute to this package. :sparkles:

- _Your name here?_ :tada:

## Maintainers

The following are the core maintainers of the `alexandra_ai_eval` package:

- [@saattrupdan](https://github.com/saattrupdan) (Dan Saattrup Nielsen; saattrupdan@alexandra.dk)
- [@AJDERS](https://github.com/AJDERS) (Anders Jess Pedersen; anders.j.pedersen@alexandra.dk)

## Project structure

```
.
├── .github
│   ├── ISSUE_TEMPLATE
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── makefile
├── poetry.toml
├── pyproject.toml
├── src
│   ├── alexandra_ai_eval
│   │   ├── __init__.py
│   │   ├── automatic_speech_recognition.py
│   │   ├── cli.py
│   │   ├── co2.py
│   │   ├── config.py
│   │   ├── country_codes.py
│   │   ├── enums.py
│   │   ├── evaluator.py
│   │   ├── exceptions.py
│   │   ├── gui.py
│   │   ├── hf_hub_utils.py
│   │   ├── leaderboard_utils.py
│   │   ├── local_hf_utils.py
│   │   ├── local_pytorch_utils.py
│   │   ├── metric_configs.py
│   │   ├── model_adjustment.py
│   │   ├── model_loading.py
│   │   ├── named_entity_recognition.py
│   │   ├── question_answering.py
│   │   ├── scoring.py
│   │   ├── sequence_classification.py
│   │   ├── spacy_utils.py
│   │   ├── task.py
│   │   ├── task_configs.py
│   │   ├── task_factory.py
│   │   └── utils.py
│   └── scripts
│       ├── add_models_to_leaderboard.py
│       ├── fix_dot_env_file.py
│       └── versioning.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_cli.py
    ├── test_co2.py
    ├── test_config.py
    ├── test_country_codes.py
    ├── test_enums.py
    ├── test_evaluator.py
    ├── test_exceptions.py
    ├── test_gui.py
    ├── test_hf_hub_utils.py
    ├── test_leaderboard_utils.py
    ├── test_local_hf_utils.py
    ├── test_local_pytorch_utils.py
    ├── test_metric_configs.py
    ├── test_model_adjustment.py
    ├── test_model_loading.py
    ├── test_named_entity_recognition.py
    ├── test_question_answering.py
    ├── test_scoring.py
    ├── test_sequence_classification.py
    ├── test_spacy_utils.py
    ├── test_task.py
    ├── test_task_configs.py
    ├── test_task_factory.py
    └── test_utils.py
```
