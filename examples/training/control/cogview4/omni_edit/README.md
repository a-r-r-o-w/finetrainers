# CogView4 Edit Control training

To launch training, you can run the following from the root directory of the repository.

```bash
chmod +x ./examples/training/sft/cogview4/omni_edit/train.sh
./examples/training/sft/cogview4/omni_edit/train.sh
```

The script should automatically download the validation dataset, but in case that doesn't happen, please make sure that a folder named `validation_dataset` exists in `examples/training/sft/cogview4/omni_edit/` and contains the validation dataset. You can also configure `validation.json` in the same directory however you like for your own validation dataset.

```bash
cd examples/training/sft/cogview4/omni_edit/
huggingface-cli download --repo-type dataset finetrainers/OmniEdit-validation-dataset --local-dir validation_dataset
```
