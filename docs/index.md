

Finetrainers is a work-in-progress library to support (accessible) training of diffusion models and various commonly used training algorithms.

# Features

DDP, FSDP-2 & HSDP support for all models
LoRA and full-rank finetuning; Conditional Control training
Memory-efficient single-GPU training
Multiple attention backends supported - flash, flex, sage, xformers (see attention docs)
Auto-detection of commonly used dataset formats
Combined image/video datasets, multiple chainable local/remote datasets, multi-resolution bucketing & more
Memory-efficient precomputation support with/without on-the-fly precomputation for large scale datasets
Standardized model specification format for training arbitrary models
Fake FP8 training (QAT upcoming!)