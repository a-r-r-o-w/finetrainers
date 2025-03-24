# Finetrainers training documentation

This directory contains the training-related specifications for all the models we support in `finetrainers`. Each model page has:
- an example training command
- inference example
- numbers on memory consumption

By default, we don't include any validation-related arguments in the example training commands. To enable validation inference every 500 steps, one can add the following arguments:

```diff
+ --validation_dataset_file <Path to a CSV/JSON/PARQUET/ARROW> \
+ --validation_steps 500
```

Arguments for training are documented in the code. For more information, please run `python train.py --help`.

## Support matrix

The following table shows the algorithms supported for training and the models they are supported for:

| Model                                     | SFT | Control | ControlNet | Distillation |
|:-----------------------------------------:|:---:|:-------:|:----------:|:------------:|
| [CogVideoX](./cogvideox.md)               | 🤗 | 😡 | 😡 | 😡 |
| [CogView4](./cogview4.md)                 | 🤗 | 😡 | 😡 | 😡 |
| [HunyuanVideo](./hunyuan_video.md)        | 🤗 | 😡 | 😡 | 😡 |
| [LTX-Video](./ltx_video.md)               | 🤗 | 😡 | 😡 | 😡 |
| [Wan](./wan.md)                           | 🤗 | 😡 | 😡 | 😡 |

For launching SFT Training:
- `--training_type lora`: Trains a new set of low-rank weights of the model, yielding a smaller adapter model. Currently, only LoRA is supported from [🤗 PEFT](https://github.com/huggingface/peft)
- `--training_type full-finetune`: Trains the full-rank weights of the model, yielding a full-parameter trained model.

Any model architecture loadable in diffusers/transformers for above models can be used for training. For example, [SkyReels-T2V](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V) is a finetune of HunyuanVideo, which is compatible for continual training out-of-the-box. Custom models can be loaded either by writing your own [ModelSpecification](TODO(aryan): add link) or by using the following set of arguments:
- `--tokenizer_id`, `--tokenizer_2_id`, `--tokenizer_3_id`: The tokenizers to use for training in conjunction with text encoder conditioning models.
- `--text_encoder_id`, `--text_encoder_2_id`, `--text_encoder_3_id`: The text encoder conditioning models.
- `--transformer_id`: The transformer model to use for training.
- `--vae_id`: The VAE model to use for training.

The above arguments should take care of most training scenarios. For any custom training scenarios, please use your own implementation of a `ModelSpecification`. These arguments should be used only if one wants to override the default components loaded from `--pretrained_model_name_or_path`. Similar to each of these arguments, there exists a set of `--<ARG>_dtype` argument to specify the precision of each component.

## Resuming training

To resume training, the following arguments can be used:
- `--checkpointing_steps`: The interval of training steps that should be completed after which the training state should be saved.
- `--checkpointing_limit`: The maximum number of checkpoints that should be saved at once. If the limit is reached, the oldest checkpoint is purged.
- `--resume_from_checkpoint <STEP_OR_LATEST>`: Can be an integer or the string `"latest"`. If an integer is provided, training will resume from that step if a checkpoint corresponding to it exists. If `"latest"` is provided, training will resume from the latest checkpoint in the `--output_dir`.

> [!IMPORTANT]
> The `--resume_from_checkpoint` argument is only compatible if the parallel backend and degrees of parallelism are the same from the previous training run. For example, changing `--dp_degree 2 --dp_shards 1` from past run to `--dp_degree 1 --dp_shards 2` in current run will not work.

## How do we handle `mixed_precision`?

The accelerate config files (the ones seen [here](../../accelerate_configs/)) that are being supplied while launching training should contain a field called `mixed_precision` and `accelerate` makes use of that if specified. We don't let users explicitly pass that from the CLI args because it can be confusing to have `transformer_dtype` and `mixed_precision` in the codebase.

`transformer_dtype` is the ultimate source of truth for the precision to be used when training. It will also most likely always have to be `torch.bfloat16` because:

* All models currently supported (except Cog-2b) do not work well in FP16 for inference, so training would be broken as well. This can be revisited if it makes sense to train in FP16 for other models added.
* The `accelerate` config files default to using "bf16", but modifying that would be at the risk of user and assumes they understand the significance of their changes.
