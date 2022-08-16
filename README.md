<div align='center'>
<img src="https://raw.githubusercontent.com/alexandrainst/AIAI-eval/feat/add-logo/gfx/logo.png" width="258" height="224">
</div>

# AIAI-Eval
### Evaluation of finetuned models.

______________________________________________________________________
[![PyPI Status](https://badge.fury.io/py/aiai_eval.svg)](https://pypi.org/project/aiai_eval/)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/AIAI-eval/aiai_eval.html)
[![License](https://img.shields.io/github/license/alexandrainst/AIAI-eval)](https://github.com/alexandrainst/AIAI-eval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/AIAI-eval)](https://github.com/alexandrainst/AIAI-eval/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-68%25-yellow.svg)](https://github.com/alexandrainst/AIAI-eval/tree/dev/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
$ poetry add <package-name>
```

### Auto-generate API documentation

To auto-generate API document for your project, run:

```
$ make docs
```

To view the documentation, run:

```
$ make view-docs
```

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   ├── model
│   │   └── model1.yaml
│   └── process
│       └── process1.yaml
├── data
│   ├── final
│   ├── processed
│   └── raw
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scripts
│   │   ├── fix_dot_env_file.py
│   │   └── versioning.py
│   └── aiai_eval
│       ├── __init__.py
│       └── demo.py
└── tests
    └── __init__.py
```
