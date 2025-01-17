import json
import logging
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import diffusers
import torch
import torch.backends
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, load_image, load_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm

from .args import Args, validate_args
from .constants import (
    FINETRAINERS_LOG_LEVEL,
    PRECOMPUTED_CONDITIONS_DIR_NAME,
    PRECOMPUTED_DIR_NAME,
    PRECOMPUTED_LATENTS_DIR_NAME,
)
from .dataset import BucketSampler, ImageOrVideoDatasetWithResizing, PrecomputedDataset
from .hooks import apply_layerwise_upcasting
from .models import get_config_from_model_name
from .patches import perform_peft_patches
from .state import State
from .utils.checkpointing import get_intermediate_ckpt_path, get_latest_ckpt_path_to_resume_from
from .utils.data_utils import should_perform_precomputation
from .utils.diffusion_utils import (
    get_scheduler_alphas,
    get_scheduler_sigmas,
    prepare_loss_weights,
    prepare_sigmas,
    prepare_target,
)
from .utils.file_utils import string_to_filename
from .utils.hub_utils import save_model_card
from .utils.memory_utils import free_memory, get_memory_statistics, make_contiguous
from .utils.model_utils import resolve_vae_cls_from_ckpt_path
from .utils.optimizer_utils import get_optimizer
from .utils.torch_utils import align_device_and_dtype, expand_tensor_dims, unwrap_model


logger = get_logger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)


class Trainer:
    def __init__(self, args: Args) -> None:
        validate_args(args)

        self.args = args
        self.args.seed = self.args.seed or datetime.now().year
        self.state = State()

        # Tokenizers
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None

        # Text encoders
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None

        # Denoisers
        self.transformer = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self.transformer_config = None
        self.vae_config = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()
        self._init_config_options()

        # Peform any patches needed for training
        if len(self.args.layerwise_upcasting_modules) > 0:
            perform_peft_patches()
        # TODO(aryan): handle text encoders
        # if any(["text_encoder" in component_name for component_name in self.args.layerwise_upcasting_modules]):
        #     perform_text_encoder_patches()

        self.state.model_name = self.args.model_name
        self.model_config = get_config_from_model_name(self.args.model_name, self.args.training_type)

    def prepare_dataset(self) -> None:
        # TODO(aryan): Make a background process for fetching
        logger.info("Initializing dataset and dataloader")

        self.dataset = ImageOrVideoDatasetWithResizing(
            data_root=self.args.data_root,
            caption_column=self.args.caption_column,
            video_column=self.args.video_column,
            resolution_buckets=self.args.video_resolution_buckets,
            dataset_file=self.args.dataset_file,
            id_token=self.args.id_token,
            remove_llm_prefixes=self.args.remove_common_llm_caption_prefixes,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            sampler=BucketSampler(self.dataset, batch_size=self.args.batch_size, shuffle=True),
            collate_fn=self.model_config.get("collate_fn"),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.pin_memory,
        )

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        load_components_kwargs = self._get_load_components_kwargs()
        condition_components, latent_components, diffusion_components = {}, {}, {}
        if not self.args.precompute_conditions:
            condition_components = self.model_config["load_condition_models"](**load_components_kwargs)
            latent_components = self.model_config["load_latent_models"](**load_components_kwargs)
            diffusion_components = self.model_config["load_diffusion_models"](**load_components_kwargs)

        components = {}
        components.update(condition_components)
        components.update(latent_components)
        components.update(diffusion_components)
        self._set_components(components)

        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()

    def prepare_precomputations(self) -> None:
        if not self.args.precompute_conditions:
            return

        logger.info("Initializing precomputations")

        if self.args.batch_size != 1:
            raise ValueError("Precomputation is only supported with batch size 1. This will be supported in future.")

        def collate_fn(batch):
            latent_conditions = [x["latent_conditions"] for x in batch]
            text_conditions = [x["text_conditions"] for x in batch]
            batched_latent_conditions = {}
            batched_text_conditions = {}
            for key in list(latent_conditions[0].keys()):
                if torch.is_tensor(latent_conditions[0][key]):
                    batched_latent_conditions[key] = torch.cat([x[key] for x in latent_conditions], dim=0)
                else:
                    # TODO(aryan): implement batch sampler for precomputed latents
                    batched_latent_conditions[key] = [x[key] for x in latent_conditions][0]
            for key in list(text_conditions[0].keys()):
                if torch.is_tensor(text_conditions[0][key]):
                    batched_text_conditions[key] = torch.cat([x[key] for x in text_conditions], dim=0)
                else:
                    # TODO(aryan): implement batch sampler for precomputed latents
                    batched_text_conditions[key] = [x[key] for x in text_conditions][0]
            return {"latent_conditions": batched_latent_conditions, "text_conditions": batched_text_conditions}

        cleaned_model_id = string_to_filename(self.args.pretrained_model_name_or_path)
        precomputation_dir = (
            Path(self.args.data_root) / f"{self.args.model_name}_{cleaned_model_id}_{PRECOMPUTED_DIR_NAME}"
        )
        should_precompute = should_perform_precomputation(precomputation_dir)
        if not should_precompute:
            logger.info("Precomputed conditions and latents found. Loading precomputed data.")
            self.dataloader = torch.utils.data.DataLoader(
                PrecomputedDataset(
                    data_root=self.args.data_root, model_name=self.args.model_name, cleaned_model_id=cleaned_model_id
                ),
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.pin_memory,
            )
            return

        logger.info("Precomputed conditions and latents not found. Running precomputation.")

        # At this point, no models are loaded, so we need to load and precompute conditions and latents
        condition_components = self.model_config["load_condition_models"](**self._get_load_components_kwargs())
        self._set_components(condition_components)
        self._move_components_to_device()
        self._disable_grad_for_components([self.text_encoder, self.text_encoder_2, self.text_encoder_3])

        if self.args.caption_dropout_p > 0 and self.args.caption_dropout_technique == "empty":
            logger.warning(
                "Caption dropout is not supported with precomputation yet. This will be supported in the future."
            )

        conditions_dir = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
        latents_dir = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME
        conditions_dir.mkdir(parents=True, exist_ok=True)
        latents_dir.mkdir(parents=True, exist_ok=True)

        accelerator = self.state.accelerator

        # Precompute conditions
        progress_bar = tqdm(
            range(0, (len(self.dataset) + accelerator.num_processes - 1) // accelerator.num_processes),
            desc="Precomputing conditions",
            disable=not accelerator.is_local_main_process,
        )
        index = 0
        for i, data in enumerate(self.dataset):
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            logger.debug(
                f"Precomputing conditions for batch {i + 1}/{len(self.dataset)} on process {accelerator.process_index}"
            )

            text_conditions = self.model_config["prepare_conditions"](
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                tokenizer_3=self.tokenizer_3,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                text_encoder_3=self.text_encoder_3,
                prompt=data["prompt"],
                device=accelerator.device,
                dtype=self.args.transformer_dtype,
            )
            filename = conditions_dir / f"conditions-{accelerator.process_index}-{index}.pt"
            torch.save(text_conditions, filename.as_posix())
            index += 1
            progress_bar.update(1)
        self._delete_components()

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after precomputing conditions: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        # Precompute latents
        latent_components = self.model_config["load_latent_models"](**self._get_load_components_kwargs())
        self._set_components(latent_components)
        self._move_components_to_device()
        self._disable_grad_for_components([self.vae])

        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()

        progress_bar = tqdm(
            range(0, (len(self.dataset) + accelerator.num_processes - 1) // accelerator.num_processes),
            desc="Precomputing latents",
            disable=not accelerator.is_local_main_process,
        )
        index = 0
        for i, data in enumerate(self.dataset):
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            logger.debug(
                f"Precomputing latents for batch {i + 1}/{len(self.dataset)} on process {accelerator.process_index}"
            )

            latent_conditions = self.model_config["prepare_latents"](
                vae=self.vae,
                image_or_video=data["video"].unsqueeze(0),
                device=accelerator.device,
                dtype=self.args.transformer_dtype,
                generator=self.state.generator,
                precompute=True,
            )
            filename = latents_dir / f"latents-{accelerator.process_index}-{index}.pt"
            torch.save(latent_conditions, filename.as_posix())
            index += 1
            progress_bar.update(1)
        self._delete_components()

        accelerator.wait_for_everyone()
        logger.info("Precomputation complete")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after precomputing latents: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        # Update dataloader to use precomputed conditions and latents
        self.dataloader = torch.utils.data.DataLoader(
            PrecomputedDataset(
                data_root=self.args.data_root, model_name=self.args.model_name, cleaned_model_id=cleaned_model_id
            ),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.pin_memory,
        )

    def prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        diffusion_components = self.model_config["load_diffusion_models"](**self._get_load_components_kwargs())
        self._set_components(diffusion_components)

        components = [self.text_encoder, self.text_encoder_2, self.text_encoder_3, self.vae]
        self._disable_grad_for_components(components)

        if self.args.training_type == "full-finetune":
            logger.info("Finetuning transformer with no additional parameters")
            self._enable_grad_for_components([self.transformer])
        else:
            logger.info("Finetuning transformer with PEFT parameters")
            self._disable_grad_for_components([self.transformer])

        # Layerwise upcasting must be applied before adding the LoRA adapter.
        # If we don't perform this before moving to device, we might OOM on the GPU. So, best to do it on
        # CPU for now, before support is added in Diffusers for loading and enabling layerwise upcasting directly.
        if self.args.training_type == "lora" and "transformer" in self.args.layerwise_upcasting_modules:
            apply_layerwise_upcasting(
                self.transformer,
                storage_dtype=self.args.layerwise_upcasting_storage_dtype,
                compute_dtype=self.args.transformer_dtype,
                skip_modules_pattern=self.args.layerwise_upcasting_skip_modules_pattern,
                non_blocking=True,
            )

        self._move_components_to_device()

        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
        else:
            transformer_lora_config = None

        # TODO(aryan): it might be nice to add some assertions here to make sure that lora parameters are still in fp32
        # even if layerwise upcasting. Would be nice to have a test as well

        self.register_saving_loading_hooks(transformer_lora_config)

    def register_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.state.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.state.accelerator, model),
                        type(unwrap_model(self.state.accelerator, self.transformer)),
                    ):
                        model = unwrap_model(self.state.accelerator, model)
                        if self.args.training_type == "lora":
                            transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                if self.args.training_type == "lora":
                    self.model_config["pipeline_cls"].save_lora_weights(
                        output_dir,
                        transformer_lora_layers=transformer_lora_layers_to_save,
                    )
                else:
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # In some cases, the scheduler needs to be loaded with specific config (e.g. in CogVideoX). Since we need
                    # to able to load all diffusion components from a specific checkpoint folder during validation, we need to
                    # ensure the scheduler config is serialized as well.
                    self.scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))

        def load_model_hook(models, input_dir):
            if not self.state.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.state.accelerator, model),
                        type(unwrap_model(self.state.accelerator, self.transformer)),
                    ):
                        transformer_ = unwrap_model(self.state.accelerator, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(self.state.accelerator, model).__class__}"
                        )
            else:
                transformer_cls_ = unwrap_model(self.state.accelerator, self.transformer).__class__

                if self.args.training_type == "lora":
                    transformer_ = transformer_cls_.from_pretrained(
                        self.args.pretrained_model_name_or_path, subfolder="transformer"
                    )
                    transformer_.add_adapter(transformer_lora_config)
                    lora_state_dict = self.model_config["pipeline_cls"].lora_state_dict(input_dir)
                    transformer_state_dict = {
                        f'{k.replace("transformer.", "")}': v
                        for k, v in lora_state_dict.items()
                        if k.startswith("transformer.")
                    }
                    incompatible_keys = set_peft_model_state_dict(
                        transformer_, transformer_state_dict, adapter_name="default"
                    )
                    if incompatible_keys is not None:
                        # check only for unexpected keys
                        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                        if unexpected_keys:
                            logger.warning(
                                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                                f" {unexpected_keys}. "
                            )
                else:
                    transformer_ = transformer_cls_.from_pretrained(os.path.join(input_dir, "transformer"))

        self.state.accelerator.register_save_state_pre_hook(save_model_hook)
        self.state.accelerator.register_load_state_pre_hook(load_model_hook)

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.training_type == "lora":
            cast_training_params([self.transformer], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": transformer_trainable_parameters,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in transformer_trainable_parameters)

        use_deepspeed_opt = (
            self.state.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.state.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.state.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit=self.args.use_8bit_bnb,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.state.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.state.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.state.train_steps * self.state.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.state.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.transformer, self.optimizer, self.dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.transformer, self.optimizer, self.dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.state.train_epochs = math.ceil(self.state.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.accelerator.init_trackers(tracker_name, config=self._get_training_info())

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        if self.vae_config is None:
            # If we've precomputed conditions and latents already, and are now re-using it, we will never load
            # the VAE so self.vae_config will not be set. So, we need to load it here.
            vae_cls_name = resolve_vae_cls_from_ckpt_path(
                self.args.pretrained_model_name_or_path, revision=self.args.revision, cache_dir=self.args.cache_dir
            )
            vae_config = vae_cls_name.load_config(
                self.args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.args.revision,
                cache_dir=self.args.cache_dir,
            )
            self.vae_config = FrozenDict(**vae_config)

        # In some cases, the scheduler needs to be loaded with specific config (e.g. in CogVideoX). Since we need
        # to able to load all diffusion components from a specific checkpoint folder during validation, we need to
        # ensure the scheduler config is serialized as well.
        if self.args.training_type == "full-finetune":
            self.scheduler.save_pretrained(os.path.join(self.args.output_dir, "scheduler"))

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
            output_dir=self.args.output_dir,
        )
        if resume_from_checkpoint_path:
            self.state.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        scheduler_sigmas = get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=accelerator.device, dtype=torch.float32)
            if scheduler_sigmas is not None
            else None
        )
        scheduler_alphas = get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=accelerator.device, dtype=torch.float32)
            if scheduler_alphas is not None
            else None
        )

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.transformer.train()
            models_to_accumulate = [self.transformer]
            epoch_loss = 0.0
            num_loss_updates = 0

            for step, batch in enumerate(self.dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    if not self.args.precompute_conditions:
                        videos = batch["videos"]
                        prompts = batch["prompts"]
                        batch_size = len(prompts)

                        if self.args.caption_dropout_technique == "empty":
                            if random.random() < self.args.caption_dropout_p:
                                prompts = [""] * batch_size

                        latent_conditions = self.model_config["prepare_latents"](
                            vae=self.vae,
                            image_or_video=videos,
                            patch_size=self.transformer_config.patch_size,
                            patch_size_t=self.transformer_config.patch_size_t,
                            device=accelerator.device,
                            dtype=self.args.transformer_dtype,
                            generator=self.state.generator,
                        )
                        text_conditions = self.model_config["prepare_conditions"](
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder,
                            tokenizer_2=self.tokenizer_2,
                            text_encoder_2=self.text_encoder_2,
                            prompt=prompts,
                            device=accelerator.device,
                            dtype=self.args.transformer_dtype,
                        )
                    else:
                        latent_conditions = batch["latent_conditions"]
                        text_conditions = batch["text_conditions"]
                        latent_conditions["latents"] = DiagonalGaussianDistribution(
                            latent_conditions["latents"]
                        ).sample(self.state.generator)

                        # This method should only be called for precomputed latents.
                        # TODO(aryan): rename this in separate PR
                        latent_conditions = self.model_config["post_latent_preparation"](
                            vae_config=self.vae_config,
                            patch_size=self.transformer_config.patch_size,
                            patch_size_t=self.transformer_config.patch_size_t,
                            **latent_conditions,
                        )
                        align_device_and_dtype(latent_conditions, accelerator.device, self.args.transformer_dtype)
                        align_device_and_dtype(text_conditions, accelerator.device, self.args.transformer_dtype)
                        batch_size = latent_conditions["latents"].shape[0]

                    latent_conditions = make_contiguous(latent_conditions)
                    text_conditions = make_contiguous(text_conditions)

                    if self.args.caption_dropout_technique == "zero":
                        if random.random() < self.args.caption_dropout_p:
                            text_conditions["prompt_embeds"].fill_(0)
                            text_conditions["prompt_attention_mask"].fill_(False)

                            # TODO(aryan): refactor later
                            if "pooled_prompt_embeds" in text_conditions:
                                text_conditions["pooled_prompt_embeds"].fill_(0)

                    sigmas = prepare_sigmas(
                        scheduler=self.scheduler,
                        sigmas=scheduler_sigmas,
                        batch_size=batch_size,
                        num_train_timesteps=self.scheduler.config.num_train_timesteps,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                        flow_logit_mean=self.args.flow_logit_mean,
                        flow_logit_std=self.args.flow_logit_std,
                        flow_mode_scale=self.args.flow_mode_scale,
                        device=accelerator.device,
                        generator=self.state.generator,
                    )
                    timesteps = (sigmas * 1000.0).long()

                    noise = torch.randn(
                        latent_conditions["latents"].shape,
                        generator=self.state.generator,
                        device=accelerator.device,
                        dtype=self.args.transformer_dtype,
                    )
                    sigmas = expand_tensor_dims(sigmas, ndim=noise.ndim)

                    # TODO(aryan): We probably don't need calculate_noisy_latents because we can determine the type of
                    # scheduler and calculate the noisy latents accordingly. Look into this later.
                    if "calculate_noisy_latents" in self.model_config.keys():
                        noisy_latents = self.model_config["calculate_noisy_latents"](
                            scheduler=self.scheduler,
                            noise=noise,
                            latents=latent_conditions["latents"],
                            timesteps=timesteps,
                        )
                    else:
                        # Default to flow-matching noise addition
                        noisy_latents = (1.0 - sigmas) * latent_conditions["latents"] + sigmas * noise
                        noisy_latents = noisy_latents.to(latent_conditions["latents"].dtype)

                    latent_conditions.update({"noisy_latents": noisy_latents})

                    weights = prepare_loss_weights(
                        scheduler=self.scheduler,
                        alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                        sigmas=sigmas,
                        flow_weighting_scheme=self.args.flow_weighting_scheme,
                    )
                    weights = expand_tensor_dims(weights, noise.ndim)

                    pred = self.model_config["forward_pass"](
                        transformer=self.transformer,
                        scheduler=self.scheduler,
                        timesteps=timesteps,
                        **latent_conditions,
                        **text_conditions,
                    )
                    target = prepare_target(
                        scheduler=self.scheduler, noise=noise, latents=latent_conditions["latents"]
                    )

                    loss = weights.float() * (pred["latents"].float() - target.float()).pow(2)
                    # Average loss across all but batch dimension
                    loss = loss.mean(list(range(1, loss.ndim)))
                    # Average loss across batch dimension
                    loss = loss.mean()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Checkpointing
                    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                        if global_step % self.args.checkpointing_steps == 0:
                            save_path = get_intermediate_ckpt_path(
                                checkpointing_limit=self.args.checkpointing_limit,
                                step=global_step,
                                output_dir=self.args.output_dir,
                            )
                            accelerator.save_state(save_path)

                    # Maybe run validation
                    should_run_validation = (
                        self.args.validation_every_n_steps is not None
                        and global_step % self.args.validation_every_n_steps == 0
                    )
                    if should_run_validation:
                        self.validate(global_step)

                loss_item = loss.detach().item()
                epoch_loss += loss_item
                num_loss_updates += 1
                logs["step_loss"] = loss_item
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    break

            if num_loss_updates > 0:
                epoch_loss /= num_loss_updates
            accelerator.log({"epoch_loss": epoch_loss}, step=global_step)
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            # Maybe run validation
            should_run_validation = (
                self.args.validation_every_n_epochs is not None
                and (epoch + 1) % self.args.validation_every_n_epochs == 0
            )
            if should_run_validation:
                self.validate(global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            transformer = unwrap_model(accelerator, self.transformer)

            if self.args.training_type == "lora":
                transformer_lora_layers = get_peft_model_state_dict(transformer)

                self.model_config["pipeline_cls"].save_lora_weights(
                    save_directory=self.args.output_dir,
                    transformer_lora_layers=transformer_lora_layers,
                )
            else:
                transformer.save_pretrained(os.path.join(self.args.output_dir, "transformer"))
        accelerator.wait_for_everyone()
        self.validate(step=global_step, final_validation=True)

        if accelerator.is_main_process:
            if self.args.push_to_hub:
                upload_folder(
                    repo_id=self.state.repo_id, folder_path=self.args.output_dir, ignore_patterns=["checkpoint-*"]
                )

        self._delete_components()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int, final_validation: bool = False) -> None:
        logger.info("Starting validation")

        accelerator = self.state.accelerator
        num_validation_samples = len(self.args.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.transformer.eval()

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        pipeline = self._get_and_prepare_pipeline_for_validation(final_validation=final_validation)

        all_processes_artifacts = []
        prompts_to_filenames = {}
        for i in range(num_validation_samples):
            # Skip current validation on all processes but one
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            prompt = self.args.validation_prompts[i]
            image = self.args.validation_images[i]
            video = self.args.validation_videos[i]
            height = self.args.validation_heights[i]
            width = self.args.validation_widths[i]
            num_frames = self.args.validation_num_frames[i]
            frame_rate = self.args.validation_frame_rate
            if image is not None:
                image = load_image(image)
            if video is not None:
                video = load_video(video)

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.model_config["validation"](
                pipeline=pipeline,
                prompt=prompt,
                image=image,
                video=video,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_videos_per_prompt=self.args.num_validation_videos_per_prompt,
                generator=torch.Generator(device=accelerator.device).manual_seed(
                    self.args.seed if self.args.seed is not None else 0
                ),
                # todo support passing `fps` for supported pipelines.
            )

            prompt_filename = string_to_filename(prompt)[:25]
            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                if artifact_value:
                    artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for index, (key, value) in enumerate(list(artifacts.items())):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = "validation-" if not final_validation else "final-"
                filename += f"{step}-{accelerator.process_index}-{index}-{prompt_filename}.{extension}"
                if accelerator.is_main_process and extension == "mp4":
                    prompts_to_filenames[prompt] = filename
                filename = os.path.join(self.args.output_dir, filename)

                if artifact_type == "image" and artifact_value:
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video" and artifact_value:
                    logger.debug(f"Saving video to {filename}")
                    # TODO: this should be configurable here as well as in validation runs where we call the pipeline that has `fps`.
                    export_to_video(artifact_value, filename, fps=frame_rate)
                    artifact_value = wandb.Video(filename, caption=prompt)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "final" if final_validation else "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    artifact_log_dict = {}

                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    if len(image_artifacts) > 0:
                        artifact_log_dict["images"] = image_artifacts
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    if len(video_artifacts) > 0:
                        artifact_log_dict["videos"] = video_artifacts
                    tracker.log({tracker_key: artifact_log_dict}, step=step)

            if self.args.push_to_hub and final_validation:
                video_filenames = list(prompts_to_filenames.values())
                prompts = list(prompts_to_filenames.keys())
                save_model_card(
                    args=self.args,
                    repo_id=self.state.repo_id,
                    videos=video_filenames,
                    validation_prompts=prompts,
                )

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline

        accelerator.wait_for_everyone()

        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        if not final_validation:
            self.transformer.train()

    def evaluate(self) -> None:
        raise NotImplementedError("Evaluation has not been implemented yet.")

    def _init_distributed(self) -> None:
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=FINETRAINERS_LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized FineTrainers")
        logger.info(self.state.accelerator.state, main_process_only=False)

    def _init_directories_and_repositories(self) -> None:
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = Path(self.args.output_dir)

            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(token=self.args.hub_token, repo_id=repo_id, exist_ok=True).repo_id

    def _init_config_options(self) -> None:
        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def _move_components_to_device(self):
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self.state.accelerator.device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2 = self.text_encoder_2.to(self.state.accelerator.device)
        if self.text_encoder_3 is not None:
            self.text_encoder_3 = self.text_encoder_3.to(self.state.accelerator.device)
        if self.transformer is not None:
            self.transformer = self.transformer.to(self.state.accelerator.device)
        if self.unet is not None:
            self.unet = self.unet.to(self.state.accelerator.device)
        if self.vae is not None:
            self.vae = self.vae.to(self.state.accelerator.device)

    def _get_load_components_kwargs(self) -> Dict[str, Any]:
        load_component_kwargs = {
            "text_encoder_dtype": self.args.text_encoder_dtype,
            "text_encoder_2_dtype": self.args.text_encoder_2_dtype,
            "text_encoder_3_dtype": self.args.text_encoder_3_dtype,
            "transformer_dtype": self.args.transformer_dtype,
            "vae_dtype": self.args.vae_dtype,
            "shift": self.args.flow_shift,
            "revision": self.args.revision,
            "cache_dir": self.args.cache_dir,
        }
        if self.args.pretrained_model_name_or_path is not None:
            load_component_kwargs["model_id"] = self.args.pretrained_model_name_or_path
        return load_component_kwargs

    def _set_components(self, components: Dict[str, Any]) -> None:
        # Set models
        self.tokenizer = components.get("tokenizer", self.tokenizer)
        self.tokenizer_2 = components.get("tokenizer_2", self.tokenizer_2)
        self.tokenizer_3 = components.get("tokenizer_3", self.tokenizer_3)
        self.text_encoder = components.get("text_encoder", self.text_encoder)
        self.text_encoder_2 = components.get("text_encoder_2", self.text_encoder_2)
        self.text_encoder_3 = components.get("text_encoder_3", self.text_encoder_3)
        self.transformer = components.get("transformer", self.transformer)
        self.unet = components.get("unet", self.unet)
        self.vae = components.get("vae", self.vae)
        self.scheduler = components.get("scheduler", self.scheduler)

        # Set configs
        self.transformer_config = self.transformer.config if self.transformer is not None else self.transformer_config
        self.vae_config = self.vae.config if self.vae is not None else self.vae_config

    def _delete_components(self) -> None:
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        self.transformer = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        free_memory()
        torch.cuda.synchronize(self.state.accelerator.device)

    def _get_and_prepare_pipeline_for_validation(self, final_validation: bool = False) -> DiffusionPipeline:
        accelerator = self.state.accelerator
        if not final_validation:
            pipeline = self.model_config["initialize_pipeline"](
                model_id=self.args.pretrained_model_name_or_path,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                tokenizer_2=self.tokenizer_2,
                text_encoder_2=self.text_encoder_2,
                transformer=unwrap_model(accelerator, self.transformer),
                vae=self.vae,
                device=accelerator.device,
                revision=self.args.revision,
                cache_dir=self.args.cache_dir,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                is_training=True,
            )
        else:
            self._delete_components()

            # Load the transformer weights from the final checkpoint if performing full-finetune
            transformer = None
            if self.args.training_type == "full-finetune":
                transformer = self.model_config["load_diffusion_models"](model_id=self.args.output_dir)["transformer"]

            pipeline = self.model_config["initialize_pipeline"](
                model_id=self.args.pretrained_model_name_or_path,
                transformer=transformer,
                device=accelerator.device,
                revision=self.args.revision,
                cache_dir=self.args.cache_dir,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                is_training=False,
            )

            # Load the LoRA weights if performing LoRA finetuning
            if self.args.training_type == "lora":
                pipeline.load_lora_weights(self.args.output_dir)

        return pipeline

    def _disable_grad_for_components(self, components: List[torch.nn.Module]):
        for component in components:
            if component is not None:
                component.requires_grad_(False)

    def _enable_grad_for_components(self, components: List[torch.nn.Module]):
        for component in components:
            if component is not None:
                component.requires_grad_(True)

    def _get_training_info(self) -> dict:
        args = self.args.to_dict()

        training_args = args.get("training_arguments", {})
        training_type = training_args.get("training_type", "")

        # LoRA/non-LoRA stuff.
        if training_type == "full-finetune":
            filtered_training_args = {
                k: v for k, v in training_args.items() if k not in {"rank", "lora_alpha", "target_modules"}
            }
        else:
            filtered_training_args = training_args

        # Diffusion/flow stuff.
        diffusion_args = args.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args

        # Rest of the stuff.
        updated_training_info = args.copy()
        updated_training_info["training_arguments"] = filtered_training_args
        updated_training_info["diffusion_arguments"] = filtered_diffusion_args
        return updated_training_info
