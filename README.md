<div align='center'>
<img src="https://raw.githubusercontent.com/alexandrainst/AIAI-eval/main/gfx/aiai-eval-logo.png" width="auto" height="224">
</div>

### Evaluation of finetuned models.

______________________________________________________________________
[![PyPI Status](https://badge.fury.io/py/aiai_eval.svg)](https://pypi.org/project/aiai_eval/)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/AIAI-eval/aiai_eval.html)
[![License](https://img.shields.io/github/license/alexandrainst/AIAI-eval)](https://github.com/alexandrainst/AIAI-eval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/AIAI-eval)](https://github.com/alexandrainst/AIAI-eval/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-53%25-orange.svg)](https://github.com/alexandrainst/AIAI-eval/tree/dev/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Installation
To install the package simply write the following command in your favorite terminal:
```
$ pip install aiai-eval
```

## Quickstart
### Benchmarking from the Command Line
The easiest way to benchmark pretrained models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:
```
$ evaluate --model-id <model_id> --task <task>
```

Here `model_id` is the HuggingFace model ID, which can be found on the [HuggingFace
Hub](https://huggingface.co/models), and `task` is the task you want to benchmark the
model on, such as "ner" for named entity recognition. See all options by typing
```
$ evaluate --help
```

The specific model version to use can also be added after the suffix '@':
```
$ evaluate --model_id <model_id>@<commit>
```

It can be a branch name, a tag name, or a commit id. It defaults to 'main' for latest.

Multiple models and tasks can be specified by just attaching multiple arguments. Here
is an example with two models:
```
$ evaluate --model_id <model_id1> --model_id <model_id2> --task ner
```

See all the arguments and options available for the `evaluate` command by typing
```
$ evaluate --help
```

### Benchmarking from a Script
In a script, the syntax is similar to the command line interface. You simply initialise
an object of the `Evaluator` class, and call this evaluate object with your favorite
models and/or datasets:
```
>>> from aiai_eval import Evaluator
>>> evaluator = Evaluator()
>>> evaluator('<model_id>', '<task>')
```


## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── gfx
│   └── logo.png
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── aiai_eval
│   │   ├── __init__.py
│   │   ├── automatic_speech_recognition.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── evaluator.py
│   │   ├── exceptions.py
│   │   ├── hf_hub.py
│   │   ├── image_to_text.py
│   │   ├── named_entity_recognition.py
│   │   ├── question_answering.py
│   │   ├── task.py
│   │   ├── task_configs.py
│   │   ├── task_factory.py
│   │   ├── text_classification.py
│   │   └── utils.py
│   └── scripts
│       ├── fix_dot_env_file.py
│       └── versioning.py
└── tests
    └── __init__.py
```
