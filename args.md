## Parameters for finetrainers/args.py

| Parameter | Description | Type | Default Value |
|---|---|---|---|
| **model_name** | Name of the model to train. | str |  |
| **pretrained_model_name_or_path** | Path or identifier for pre-trained model from HuggingFace Hub. | str | None |
| **revision** | Revision of the identifier for pre-trained model from HuggingFace Hub. | Optional[str] | None |
| **variant** | Model file variant from HuggingFace Hub (e.g., 'fp16'). | Optional[str] | None |
| **cache_dir** | Directory where downloaded models and datasets are stored. | Optional[str] | None |
| **text_encoder_dtype** | Data type for text encoder. | torch.dtype | torch.bfloat16 |
| **text_encoder_2_dtype** | Data type for text encoder 2. | torch.dtype | torch.bfloat16 |
| **text_encoder_3_dtype** | Data type for text encoder 3. | torch.dtype | torch.bfloat16 |
| **transformer_dtype** | Data type for Transformer model. | torch.dtype | torch.bfloat16 |
| **vae_dtype** | Data type for VAE model. | torch.dtype | torch.bfloat16 |
| **data_root** | Folder containing training data. | str | None |
| **dataset_file** | Path to CSV file for loading prompts/video paths in this format. | Optional[str] | None |
| **video_column** | Column in dataset containing videos, or name of file in `--data_root` folder containing newline-separated paths to video data. | str | None |
| **caption_column** | Column in dataset containing instance prompts for each video, or name of file in `--data_root` folder containing newline-separated instance prompts. | str | None |
| **id_token** | If specified, identifier token to be prepended to each prompt. | Optional[str] | None |
| **image_resolution_buckets** | Image resolution buckets. | List[Tuple[int, int]] | None |
| **video_resolution_buckets** | Video resolution buckets. | List[Tuple[int, int, int]] | None |
| **video_reshape_mode** | All input videos will be reshaped to this mode. Choose from ['center', 'random', 'none']. | Optional[str] | None |
| **caption_dropout_p** | Probability of caption token dropout. | float | 0.00 |
| **caption_dropout_technique** | Technique to use for caption dropout. | str | "empty" |
| **precompute_conditions** | Whether to precompute model conditions. | bool | False |
| **dataloader_num_workers** | Number of subprocesses to use for data loading. 0 means data will be loaded in the main process. | int | 0 |
| **pin_memory** | Whether to use pinned memory setting in Pytorch dataloader. | bool | False |
| **flow_resolution_shifting** | Resolution-dependent shift in timestep schedule. | bool | False |
| **flow_weighting_scheme** | Uses "none" weighting scheme by default for uniform sampling and uniform loss. | str | "none" |
| **flow_logit_mean** | Mean to use when using `'logit_normal'` weighting scheme. | float | 0.0 |
| **flow_logit_std** | Standard deviation to use when using `'logit_normal'` weighting scheme. | float | 1.0 |
| **flow_mode_scale** | Scale for mode weighting scheme. Only effective when using `'mode'` as `weighting_scheme`. | float | 1.29 |
| **training_type** | Type of training to perform. Choose from ['lora']. | str | None |
| **seed** | Seed for reproducible training. | int | 42 |
| **mixed_precision** | Whether to use mixed precision. Default is current system's accelerate config value or flag passed to `accelerate.launch`. Use this argument to override accelerate config. | str | None |
| **batch_size** | Batch size for training dataloader (per device). | int | 1 |
| **train_epochs** | Number of training epochs. | int | 1 |
| **train_steps** | Total number of training steps to perform. If specified, overrides `--num_train_epochs`. | Optional[int] | None |
| **rank** | Rank of LoRA matrices. | int | 128 |
| **lora_alpha** | lora_alpha for calculating scaling factor (lora_alpha / rank) of LoRA matrices. | float | 64 |
| **target_modules** | Target modules for LoRA. | List[str] | ["to_k", "to_q", "to_v", "to_out.0"] |
| **gradient_accumulation_steps** | Number of updates steps to accumulate before performing backward/update pass. | int | 1 |
| **gradient_checkpointing** | Whether to use gradient checkpointing to save memory (slower backward pass). | bool | False |
| **checkpointing_steps** | Save a checkpoint of training state every X updates. These can serve as final checkpoints if they perform better than the last checkpoint and are suitable for resuming training using `--resume_from_checkpoint`. | int | 500 |
| **checkpointing_limit** | Maximum number of checkpoints to keep. | Optional[int] | None |
| **resume_from_checkpoint** | Whether to resume training from previous checkpoint. Use path saved by `--checkpointing_steps`, or use `"latest"` to automatically select the last available checkpoint. | Optional[str] | None |
| **enable_slicing** | Whether to use VAE slicing to save memory. | bool | False |
| **enable_tiling** | Whether to use VAE tiling to save memory. | bool | False |
| **optimizer** | Type of optimizer to use. | str | "adamw" |
| **use_8bit_bnb** | Whether to use 8-bit variant of `--optimizer` using `bitsandbytes`. | bool | False |
| **lr** | Initial learning rate (after warmup period). | float | 1e-4 |
| **scale_lr** | Scale learning rate by number of GPUs, gradient accumulation steps, and batch size. | bool | False |
| **lr_scheduler** | Type of scheduler to use. | str | "cosine_with_restarts" |
| **lr_warmup_steps** | Number of steps for lr scheduler warmup. | int | 0 |
| **lr_num_cycles** | Number of hard resets of lr in cosine_with_restarts scheduler. | int | 1 |
| **lr_power** | Power factor for polynomial scheduler. | float | 1.0 |
| **beta1** | beta1 parameter for Adam and Prodigy optimizers. | float | 0.9 |
| **beta2** | beta2 parameter for Adam and Prodigy optimizers. | float | 0.95 |
| **beta3** | Coefficient for computing step size in Prodigy optimizer (using moving average). If set to None, uses square root of beta2. | Optional[float] | 0.999 |
| **weight_decay** | Weight decay to use with optimizer. | float | 0.0001 |
| **epsilon** | Epsilon value for Adam and Prodigy optimizers. | float | 1e-8 |
| **max_grad_norm** | Maximum gradient norm. | float | 1.0 |
| **validation_prompts** | One or more prompts to use during validation to verify model is learning. Multiple validation prompts must be separated by '--validation_prompt_seperator' string. | Optional[List[str]] | None |
| **validation_images** | One or more image paths/URLs to use during validation to verify model is learning. Multiple validation paths must be separated by '--validation_prompt_seperator' string. These must correspond to the order of validation prompts. | Optional[List[str]] | None |
| **validation_videos** | One or more video paths/URLs to use during validation to verify model is learning. Multiple validation paths must be separated by '--validation_prompt_seperator' string. These must correspond to the order of validation prompts. | Optional[List[str]] | None |
| **num_validation_videos_per_prompt** | Number of videos that should be generated during validation per `validation_prompt`. | int | 1 |
| **validation_every_n_epochs** | Run validation every X training epochs. Validation consists of running validation prompts `args.num_validation_videos` times. | Optional[int] | None |
| **validation_every_n_steps** | Run validation every X training steps. Validation consists of running validation prompts `args.num_validation_videos` times. | Optional[int] | None |
| **enable_model_cpu_offload** | Whether to enable per-model CPU offload to save memory when running validation/testing. | bool | False |
| **tracker_name** | Project tracker name. | str | "finetrainers" |
| **push_to_hub** | Whether to push model to Hub. | bool | False |
| **hub_token** | Token to use for pushing to Model Hub. | Optional[str] | None |
| **hub_model_id** | Repository name to sync with local `output_dir`. | Optional[str] | None |
| **output_dir** | Output directory where model predictions and checkpoints will be written. | str | "finetrainer-training" |
| **logging_dir** | Directory where logs will be saved. | Optional[str] | "logs" |
| **allow_tf32** | Whether to allow TF32 on Ampere GPUs. Can be used to speed up training. See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices for more details. | bool | False |
| **nccl_timeout** | Maximum timeout in seconds for allgather or related operations to fail in multi-GPU/multi-node training setup. | int | 1800 |
| **report_to** | Integration to report results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"`, and `"comet_ml"`. Use `"all"` to report to all integrations. | str | "wandb" |