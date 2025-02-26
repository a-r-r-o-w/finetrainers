# How to contribute to Finetrainers

Finetrainers is an early-stage library for training diffusion models. Everyone is welcome to contribute - models, algorithms, refactors, docs, etc. - but due to the early stage of the project, we recommend bigger contributions be discussed in an issue before submitting a PR. Eventually, we will have a better process for this!

## How to contribute

### Adding a new model

If you would like to add a new model, please follow these steps:

- Create a new file in the `finetrainers/models` directory with the model name (if it's new), or use the same directory if it's a variant of an existing model.
- Implement the model specification in the file. For more details on what a model specification should look like, see the [ModelSpecification](TODO(aryan): add link) documentation.
- Update the supported configs in `finetrainers/config.py` to include the new model and the training types supported.
- Add a dummy model specification in the `tests/models` directory.
- Make sure to test training with the following settings:
  - Single GPU
  - 2x GPU with `--dp_degree 2 --dp_shards 1`
  - 2x GPU with `--dp_degree 1 --dp_shards 2`
  
  For `SFTTrainer` additions, please make sure to train with atleast 1000 steps (atleast 2000 data points) to ensure the model training is working as expected.
- Open a PR with your changes. Please make sure to share your wandb logs for the above training settings in the PR description. This will help us verify the training is working as expected.

### Adding a new algorithm

Currently, we are not accepting algorithm contributions. We will update this section once we are better ready 🤗

### Refactors

The library is in a very early stage. There are many instances of dead code, poorly written abstractions, and other issues. If you would like to refactor/clean-up a part of the codebase, please open an issue to discuss the changes before submitting a PR.

### Dataset improvements

Any changes to dataset/dataloader implementations can be submitted directly. The improvements and reasons for the changes should be conveyed appropriately for us to move quickly 🤗

### Documentation

Due to the early stage of the project, the documentation is not as comprehensive as we would like. Any improvements/refactors are welcome directly!

## Asking for help

If you have any questions, feel free to open an issue and we will be sure to help you out asap! Please make sure to describe your issues in either English (preferable) or Chinese. Any other language will make it hard for us to help you, so we will most likely close such issues without explanation/answer.
