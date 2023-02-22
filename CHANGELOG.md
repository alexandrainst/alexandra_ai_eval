# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2023-02-22

### Added

- Support for evaluation of local Hugging Face models.
- Tests for the `question_answering`-task.
- The `automatic_speech_recognition`-task.
- Util functions, `leaderboard_utils`, for interacting with the associated REST-api which interacts with the leaderboard holding the evaluation results.
- A new function in the `evaluator` module, called `_send_results_to_leaderboard` which sends evaluation results to the leaderboard using the util functions from `leaderboard_utils`, and tests for this function and `leaderboard_utils`.
- The `discourse-coherence`-task.
- Support for integer labels.

## [v0.0.1] - 2022-08-29

### Added

- First release, which includes evaluation of sentiment models from the Hugging Face
  Hub. This can be run with the CLI using the `evaluate` command, or via a script using
  the `Evaluator` class.
