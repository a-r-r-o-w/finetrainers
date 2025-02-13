import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets.distributed
import diffusers
import torch
import torch.backends
import transformers
import wandb
from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig
from tqdm import tqdm

from . import data, optimizer, parallel, utils
from .args import Args, validate_args
from .hooks import apply_layerwise_upcasting
from .logging import logger
from .models import ModelSpecification, get_model_specifiction_cls
from .patches import perform_peft_patches
from .processors import Processor, ProcessorType, get_condition
from .state import State, TrainState


class Trainer:
    def __init__(self, args: Args) -> None:
        validate_args(args)

        self.args = args
        self.state = State()
        self.state.train_state = TrainState()

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

        # Optimizer & LR scheduler
        self.optimizer = None
        self.lr_scheduler = None

        # Trainer-specific conditions
        self.caption_preprocessing_conditions: List[Processor] = []
        self.caption_postprocessing_conditions: List[Processor] = []

        self.state.condition_types = self.args.conditions

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()
        self._init_config_options()
        self._init_non_model_conditions()

        model_specification_cls = get_model_specifiction_cls(self.args.model_name, self.args.training_type)
        self.model_specification: ModelSpecification = model_specification_cls(
            pretrained_model_name_or_path=self.args.pretrained_model_name_or_path,
            tokenizer_id=self.args.tokenizer_id,
            tokenizer_2_id=self.args.tokenizer_2_id,
            tokenizer_3_id=self.args.tokenizer_3_id,
            text_encoder_id=self.args.text_encoder_id,
            text_encoder_2_id=self.args.text_encoder_2_id,
            text_encoder_3_id=self.args.text_encoder_3_id,
            transformer_id=self.args.transformer_id,
            vae_id=self.args.vae_id,
            text_encoder_dtype=self.args.text_encoder_dtype,
            text_encoder_2_dtype=self.args.text_encoder_2_dtype,
            text_encoder_3_dtype=self.args.text_encoder_3_dtype,
            transformer_dtype=self.args.transformer_dtype,
            vae_dtype=self.args.vae_dtype,
            revision=self.args.revision,
            cache_dir=self.args.cache_dir,
        )

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        # TODO(aryan): allow configurability
        parallel_state = self.state.parallel_state
        dataset = data.initialize_dataset(self.args.data_root, dataset_type="video", streaming=True, infinite=True)
        dataset._data = datasets.distributed.split_dataset_by_node(
            dataset._data, parallel_state.rank, parallel_state._world_size
        )
        self.dataset = dataset

        # TODO(aryan): support batch size > 1
        self.dataloader = data.DPDataLoader(
            parallel_state.rank, dataset, batch_size=1, num_workers=self.args.dataloader_num_workers
        )

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        condition_components, latent_components, diffusion_components = {}, {}, {}
        if not self.args.precompute_conditions:
            condition_components = self.model_specification.load_condition_models(self.state.condition_types)
            latent_components = self.model_specification.load_latent_models()
            diffusion_components = self.model_specification.load_diffusion_models()

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

        if self.state.parallel_state.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )

    def prepare_precomputations(self) -> None:
        if not self.args.precompute_conditions:
            return

        # logger.info("Initializing precomputations")

        # if self.args.batch_size != 1:
        #     raise ValueError("Precomputation is only supported with batch size 1. This will be supported in future.")

        # def collate_fn(batch):
        #     latent_model_conditions = [x["latent_model_conditions"] for x in batch]
        #     condition_model_conditions = [x["condition_model_conditions"] for x in batch]
        #     batched_latent_model_conditions = {}
        #     batched_condition_model_conditions = {}
        #     for key in list(latent_model_conditions[0].keys()):
        #         if torch.is_tensor(latent_model_conditions[0][key]):
        #             batched_latent_model_conditions[key] = torch.cat([x[key] for x in latent_model_conditions], dim=0)
        #         else:
        #             # TODO(aryan): implement batch sampler for precomputed latents
        #             batched_latent_model_conditions[key] = [x[key] for x in latent_model_conditions][0]
        #     for key in list(condition_model_conditions[0].keys()):
        #         if torch.is_tensor(condition_model_conditions[0][key]):
        #             batched_condition_model_conditions[key] = torch.cat(
        #                 [x[key] for x in condition_model_conditions], dim=0
        #             )
        #         else:
        #             # TODO(aryan): implement batch sampler for precomputed latents
        #             batched_condition_model_conditions[key] = [x[key] for x in condition_model_conditions][0]
        #     return {
        #         "latent_model_conditions": batched_latent_model_conditions,
        #         "condition_model_conditions": batched_condition_model_conditions,
        #     }

        # cleaned_model_id = utils.string_to_filename(self.args.pretrained_model_name_or_path)
        # precomputation_dir = (
        #     Path(self.args.data_root) / f"{self.args.model_name}_{cleaned_model_id}_{PRECOMPUTED_DIR_NAME}"
        # )
        # should_precompute = utils.should_perform_precomputation(precomputation_dir)
        # if not should_precompute:
        #     logger.info("Precomputed conditions and latents found. Loading precomputed data.")
        #     self.dataloader = torch.utils.data.DataLoader(
        #         PrecomputedDataset(
        #             data_root=self.args.data_root, model_name=self.args.model_name, cleaned_model_id=cleaned_model_id
        #         ),
        #         batch_size=self.args.batch_size,
        #         shuffle=True,
        #         collate_fn=collate_fn,
        #         num_workers=self.args.dataloader_num_workers,
        #         pin_memory=self.args.pin_memory,
        #     )
        #     return

        # logger.info("Precomputed conditions and latents not found. Running precomputation.")

        # # At this point, no models are loaded, so we need to load and precompute conditions and latents
        # condition_components = self.model_specification.load_condition_models(self.state.condition_types)
        # self._set_components(condition_components)
        # self._move_components_to_device()
        # self._disable_grad_for_components([self.text_encoder, self.text_encoder_2, self.text_encoder_3])

        # if self.args.caption_dropout_p > 0 and self.args.caption_dropout_technique == "empty":
        #     logger.warning(
        #         "Caption dropout is not supported with precomputation yet. This will be supported in the future."
        #     )

        # conditions_dir = precomputation_dir / PRECOMPUTED_CONDITIONS_DIR_NAME
        # latents_dir = precomputation_dir / PRECOMPUTED_LATENTS_DIR_NAME
        # conditions_dir.mkdir(parents=True, exist_ok=True)
        # latents_dir.mkdir(parents=True, exist_ok=True)

        # accelerator = self.state.accelerator

        # # Precompute conditions
        # progress_bar = tqdm(
        #     range(0, (len(self.dataset) + accelerator.num_processes - 1) // accelerator.num_processes),
        #     desc="Precomputing conditions",
        #     disable=not accelerator.is_local_main_process,
        # )
        # index = 0
        # for i, data in enumerate(self.dataset):
        #     if i % accelerator.num_processes != accelerator.process_index:
        #         continue

        #     logger.debug(
        #         f"Precomputing conditions for batch {i + 1}/{len(self.dataset)} on process {accelerator.process_index}"
        #     )

        #     condition_model_conditions = self.model_specification.prepare_conditions(
        #         tokenizer=self.tokenizer,
        #         tokenizer_2=self.tokenizer_2,
        #         tokenizer_3=self.tokenizer_3,
        #         text_encoder=self.text_encoder,
        #         text_encoder_2=self.text_encoder_2,
        #         text_encoder_3=self.text_encoder_3,
        #         **data,
        #     )
        #     filename = conditions_dir / f"conditions-{accelerator.process_index}-{index}.pt"
        #     torch.save(condition_model_conditions, filename.as_posix())
        #     index += 1
        #     progress_bar.update(1)
        # self._delete_components()

        # memory_statistics = utils.get_memory_statistics()
        # logger.info(f"Memory after precomputing conditions: {json.dumps(memory_statistics, indent=4)}")
        # torch.cuda.reset_peak_memory_stats(accelerator.device)

        # # Precompute latents
        # latent_components = self.model_specification.load_latent_models()
        # self._set_components(latent_components)
        # self._move_components_to_device()
        # self._disable_grad_for_components([self.vae])

        # if self.vae is not None:
        #     if self.args.enable_slicing:
        #         self.vae.enable_slicing()
        #     if self.args.enable_tiling:
        #         self.vae.enable_tiling()

        # progress_bar = tqdm(
        #     range(0, (len(self.dataset) + accelerator.num_processes - 1) // accelerator.num_processes),
        #     desc="Precomputing latents",
        #     disable=not accelerator.is_local_main_process,
        # )
        # index = 0
        # for i, data in enumerate(self.dataset):
        #     if i % accelerator.num_processes != accelerator.process_index:
        #         continue

        #     logger.debug(
        #         f"Precomputing latents for batch {i + 1}/{len(self.dataset)} on process {accelerator.process_index}"
        #     )

        #     image_or_video_key = "video" if "video" in data else "image"
        #     image_or_video = data[image_or_video_key].unsqueeze(0)
        #     latent_conditions = self.model_specification.prepare_latents(
        #         vae=self.vae,
        #         image_or_video=image_or_video,
        #         generator=self.state.generator,
        #         precompute=True,
        #     )
        #     filename = latents_dir / f"latents-{accelerator.process_index}-{index}.pt"
        #     torch.save(latent_conditions, filename.as_posix())
        #     index += 1
        #     progress_bar.update(1)
        # self._delete_components()

        # accelerator.wait_for_everyone()
        # logger.info("Precomputation complete")

        # memory_statistics = utils.get_memory_statistics()
        # logger.info(f"Memory after precomputing latents: {json.dumps(memory_statistics, indent=4)}")
        # torch.cuda.reset_peak_memory_stats(accelerator.device)

        # # Update dataloader to use precomputed conditions and latents
        # self.dataloader = torch.utils.data.DataLoader(
        #     PrecomputedDataset(
        #         data_root=self.args.data_root, model_name=self.args.model_name, cleaned_model_id=cleaned_model_id
        #     ),
        #     batch_size=self.args.batch_size,
        #     shuffle=True,
        #     collate_fn=collate_fn,
        #     num_workers=self.args.dataloader_num_workers,
        #     pin_memory=self.args.pin_memory,
        # )

    def register_saving_loading_hooks(self, transformer_lora_config):
        pass
        # # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # def save_model_hook(models, weights, output_dir):
        #     if self.state.accelerator.is_main_process:
        #         transformer_lora_layers_to_save = None

        #         for model in models:
        #             if isinstance(
        #                 utils.unwrap_model(self.state.accelerator, model),
        #                 type(utils.unwrap_model(self.state.accelerator, self.transformer)),
        #             ):
        #                 model = utils.unwrap_model(self.state.accelerator, model)
        #                 if self.args.training_type == "lora":
        #                     transformer_lora_layers_to_save = get_peft_model_state_dict(model)
        #             else:
        #                 raise ValueError(f"Unexpected save model: {model.__class__}")

        #             # make sure to pop weight so that corresponding model is not saved again
        #             if weights:
        #                 weights.pop()

        #         if self.args.training_type == "lora":
        #             self.model_specification.save_lora_weights(output_dir, transformer_lora_layers_to_save)
        #         else:
        #             self.model_specification.save_model(output_dir, transformer=model, scheduler=self.scheduler)

        # def load_model_hook(models, input_dir):
        #     if not self.state.accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         while len(models) > 0:
        #             model = models.pop()
        #             if isinstance(
        #                 utils.unwrap_model(self.state.accelerator, model),
        #                 type(utils.unwrap_model(self.state.accelerator, self.transformer)),
        #             ):
        #                 transformer_ = utils.unwrap_model(self.state.accelerator, model)
        #             else:
        #                 raise ValueError(
        #                     f"Unexpected save model: {utils.unwrap_model(self.state.accelerator, model).__class__}"
        #                 )
        #     else:
        #         transformer_cls_ = utils.unwrap_model(self.state.accelerator, self.transformer).__class__

        #         if self.args.training_type == "lora":
        #             transformer_ = transformer_cls_.from_pretrained(
        #                 self.args.pretrained_model_name_or_path, subfolder="transformer"
        #             )
        #             transformer_.add_adapter(transformer_lora_config)
        #             lora_state_dict = self.model_specification.pipeline_cls.lora_state_dict(input_dir)
        #             transformer_state_dict = {
        #                 f'{k.replace("transformer.", "")}': v
        #                 for k, v in lora_state_dict.items()
        #                 if k.startswith("transformer.")
        #             }
        #             incompatible_keys = set_peft_model_state_dict(
        #                 transformer_, transformer_state_dict, adapter_name="default"
        #             )
        #             if incompatible_keys is not None:
        #                 # check only for unexpected keys
        #                 unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        #                 if unexpected_keys:
        #                     logger.warning(
        #                         f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
        #                         f" {unexpected_keys}. "
        #                     )
        #         else:
        #             transformer_ = transformer_cls_.from_pretrained(os.path.join(input_dir, "transformer"))

        # self.state.accelerator.register_save_state_pre_hook(save_model_hook)
        # self.state.accelerator.register_load_state_pre_hook(load_model_hook)

    def prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        parallel_state = self.state.parallel_state

        if self.args.precompute_conditions:
            diffusion_components = self.model_specification.load_diffusion_models()
            self._set_components(diffusion_components)

        components = [self.text_encoder, self.text_encoder_2, self.text_encoder_3, self.vae]
        self._disable_grad_for_models(components)

        if self.args.training_type == "full-finetune":
            logger.info("Finetuning transformer with no additional parameters")
            self._enable_grad_for_models([self.transformer])
        else:
            logger.info("Finetuning transformer with PEFT parameters")
            self._disable_grad_for_models([self.transformer])

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

        transformer_lora_config = None
        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)

        # TODO(aryan): it might be nice to add some assertions here to make sure that lora parameters are still in fp32
        # even if layerwise upcasting. Would be nice to have a test as well
        self.register_saving_loading_hooks(transformer_lora_config)

        # Make sure the trainable params are in float32 if data sharding is not enabled. For FSDP, we need all
        # parameters to be of the same dtype.
        if self.args.training_type == "lora":
            casting_dtype = torch.float32 if not parallel_state.data_sharding_enabled else self.args.transformer_dtype
            cast_training_params([self.transformer], dtype=casting_dtype)

        # For training LoRAs, we can be a little more optimal. Currently, the OptimizerWrapper only accepts torch::nn::Module.
        # This causes us to loop over all the parameters (even ones that don't require gradients, as in LoRA) at each optimizer
        # step. This is OK (see https://github.com/pytorch/pytorch/blob/2f40f789dafeaa62c4e4b90dbf4a900ff6da2ca4/torch/optim/sgd.py#L85-L99)
        # but can be optimized a bit by maybe creating a simple wrapper module encompassing the actual parameters that require
        # gradients. TODO(aryan): look into it in the future.
        model_parts = [self.transformer]
        self.state.num_trainable_parameters = sum(
            p.numel() for m in model_parts for p in m.parameters() if p.requires_grad
        )

        # Setup distributed optimizer and lr scheduler
        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_state = TrainState()
        optim = optimizer.get_optimizer(
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        lr_scheduler = optimizer.get_lr_scheduler(
            name=self.args.lr_scheduler,
            optimizers=optim.optimizers,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
            # TODO(aryan): handle last_epoch
        )

        self.optimizer = optim
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        parallel_state = self.state.parallel_state
        world_mesh = parallel_state.get_mesh()

        if parallel_state.context_parallel_enabled:
            raise NotImplementedError(
                "Context parallelism is not supported yet. This will be supported in the future."
            )

        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            # TODO(aryan): support other checkpointing types
            utils.apply_gradient_checkpointing(self.transformer, checkpointing_type="full")

        # Enable DDP, FSDP or HSDP
        if parallel_state.data_sharding_enabled:
            if parallel_state.data_replication_enabled:
                logger.info("Applying HSDP to the model")
            else:
                logger.info("Applying FSDP to the model")

            # Apply FSDP or HSDP
            if parallel_state.data_replication_enabled or parallel_state.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            parallel.apply_fsdp2(
                model=self.transformer,
                dp_mesh=world_mesh[dp_mesh_names],
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_state.pipeline_parallel_enabled,
                cpu_offload=False,  # TODO(aryan): needs to be tested and allowed for enabling later
            )
        elif parallel_state.data_replication_enabled:
            logger.info("Applying DDP to the model")

            if world_mesh.ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")

            parallel.apply_ddp(model=self.transformer, dp_mesh=world_mesh)

        self._move_components_to_device()

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        # In some cases, the scheduler needs to be loaded with specific config (e.g. in CogVideoX). Since we need
        # to able to load all diffusion components from a specific checkpoint folder during validation, we need to
        # ensure the scheduler config is serialized as well.
        if self.args.training_type == "full-finetune":
            self.model_specification.save_model(self.args.output_dir, scheduler=self.scheduler)

        # TODO(aryan): handle per-device batch_size > 1

        global_batch_size = self.args.batch_size * self.state.parallel_state.world_size
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-device batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        train_state = self.state.train_state
        parallel_state = self.state.parallel_state
        device = parallel_state.device

        train_state.global_step = 0
        train_state.observed_data_samples = 0
        # first_epoch = 0
        initial_global_step = 0

        # TODO(aryan): handle resuming from checkpoint later
        # # Potentially load in the weights and states from a previous save
        # (
        #     resume_from_checkpoint_path,
        #     initial_global_step,
        #     global_step,
        #     first_epoch,
        # ) = utils.get_latest_ckpt_path_to_resume_from(
        #     resume_from_checkpoint=self.args.resume_from_checkpoint,
        #     num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        #     output_dir=self.args.output_dir,
        # )
        # if resume_from_checkpoint_path:
        #     self.state.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not parallel_state.is_local_main_process,
        )

        generator = torch.Generator(device=device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        scheduler_sigmas = utils.get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=device, dtype=torch.float32) if scheduler_sigmas is not None else None
        )
        scheduler_alphas = utils.get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=device, dtype=torch.float32) if scheduler_alphas is not None else None
        )

        self.transformer.train()
        data_iterator = iter(self.dataloader)

        while (
            train_state.step < self.args.train_steps and train_state.observed_data_samples < self.args.max_data_samples
        ):
            batch = next(data_iterator)
            batch_size = len(batch["caption"])

            train_state.step += 1
            # TODO(aryan): this is not correct. We need to handle PP and DP properly
            train_state.observed_data_samples += batch_size * parallel_state._world_size

            logger.debug(f"Starting training step ({train_state.step}/{self.args.train_steps})")
            self.optimizer.zero_grad()

            if not self.args.precompute_conditions:
                latent_model_conditions = self.model_specification.prepare_latents(
                    vae=self.vae,
                    generator=self.state.generator,
                    precompute=False,
                    **batch,
                )
                condition_model_conditions = self.model_specification.prepare_conditions(
                    tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2,
                    tokenizer_3=self.tokenizer_3,
                    text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2,
                    text_encoder_3=self.text_encoder_3,
                    generator=self.state.generator,
                    precompute=False,
                    **batch,
                )
            else:
                # TODO(aryan)
                raise NotImplementedError("Precomputation is not supported yet.")
                latent_model_conditions = batch["latent_model_conditions"]
                condition_model_conditions = batch["condition_model_conditions"]
                latent_model_conditions["latents"] = DiagonalGaussianDistribution(
                    latent_model_conditions["latents"]
                ).sample(self.state.generator)

            utils.align_device_and_dtype(latent_model_conditions, device, self.args.transformer_dtype)
            utils.align_device_and_dtype(condition_model_conditions, device, self.args.transformer_dtype)
            latent_model_conditions = utils.make_contiguous(latent_model_conditions)
            condition_model_conditions = utils.make_contiguous(condition_model_conditions)

            sigmas = utils.prepare_sigmas(
                scheduler=self.scheduler,
                sigmas=scheduler_sigmas,
                batch_size=batch_size,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
                flow_logit_mean=self.args.flow_logit_mean,
                flow_logit_std=self.args.flow_logit_std,
                flow_mode_scale=self.args.flow_mode_scale,
                device=device,
                generator=self.state.generator,
            )

            if parallel_state.pipeline_parallel_enabled:
                raise NotImplementedError(
                    "Pipeline parallelism is not supported yet. This will be supported in the future."
                )
            else:
                pred, target, sigmas = self.model_specification.forward(
                    transformer=self.transformer,
                    scheduler=self.scheduler,
                    condition_model_conditions=condition_model_conditions,
                    latent_model_conditions=latent_model_conditions,
                    sigmas=sigmas,
                )

                timesteps = (sigmas * 1000.0).long()
                weights = utils.prepare_loss_weights(
                    scheduler=self.scheduler,
                    alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                    sigmas=sigmas,
                    flow_weighting_scheme=self.args.flow_weighting_scheme,
                )
                weights = utils.expand_tensor_dims(weights, pred.ndim)

                loss = weights.float() * (pred.float() - target.float()).pow(2)
                # Average loss across all but batch dimension
                loss = loss.mean(list(range(1, loss.ndim)))
                # Average loss across batch dimension
                loss = loss.mean()

                del pred, target
                loss.backward()

            # Clip gradients
            model_parts = [self.transformer]
            grad_norm = utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                self.args.max_grad_norm,
                foreach=True,
                pp_mesh=self.state.pp_mesh if parallel_state.pipeline_parallel_enabled else None,
            )

            self.optimizer.step()
            self.lr_scheduler.step()

            logs = {}
            logs["grad_norm"] = grad_norm.detach().item()
            if (
                parallel_state.data_replication_enabled
                or parallel_state.data_sharding_enabled
                or parallel_state.context_parallel_enabled
            ):
                loss = loss.detach()
                dp_cp_mesh = parallel_state.get_mesh()["dp_cp"]
                global_avg_loss, global_max_loss = (
                    parallel.dist_mean(loss, dp_cp_mesh),
                    parallel.dist_max(loss, dp_cp_mesh),
                )
            else:
                global_avg_loss = global_max_loss = loss.detach().item()

            logs["global_avg_loss"] = global_avg_loss
            logs["global_max_loss"] = global_max_loss
            progress_bar.update(1)
            progress_bar.set_postfix(logs)

            if train_state.step % self.args.logging_steps == 0:
                parallel_state.log(logs, step=train_state.step)

            train_state.log_steps.append(train_state.step)
            train_state.global_avg_losses.append(global_avg_loss)
            train_state.global_max_losses.append(global_max_loss)

            # TODO(aryan): handle checkpointing
            if train_state.step % self.args.validation_steps == 0:
                self.validate(step=train_state.step, final_validation=False)

        parallel_state.wait_for_everyone()

        if parallel_state.is_main_process:
            # TODO(aryan): handle compiled models by unwrapping and exporting model
            # transformer = utils.unwrap_model(accelerator, self.transformer)

            # if self.args.training_type == "lora":
            #     transformer_lora_layers = get_peft_model_state_dict(transformer)
            #     self.model_specification.save_lora_weights(self.args.output_dir, transformer_lora_layers)
            # else:
            #     self.model_specification.save_model(self.args.output_dir, transformer=transformer)
            pass

        parallel_state.wait_for_everyone()
        self.validate(step=train_state.step, final_validation=True)

        if parallel_state.is_main_process:
            if self.args.push_to_hub:
                upload_folder(
                    repo_id=self.state.repo_id, folder_path=self.args.output_dir, ignore_patterns=["checkpoint-*"]
                )

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        parallel_state.destroy()

    def validate(self, step: int, final_validation: bool = False) -> None:
        logger.info("Starting validation")

        # 1. Load validation dataset
        parallel_state = self.state.parallel_state
        dataset = data.ValidationDataset(self.args.validation_dataset_file)
        dataset._data = datasets.distributed.split_dataset_by_node(
            dataset._data, parallel_state.rank, parallel_state.world_size
        )
        validation_dataloader = data.DPDataLoader(
            parallel_state.rank,
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items,
        )
        main_process_prompts_to_filenames = {}  # Used to save model card
        all_processes_artifacts = []  # Used to gather artifacts from all processes

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        self.transformer.eval()
        pipeline = self._get_and_prepare_pipeline_for_validation(final_validation=final_validation)

        # 2. Run validation
        for validation_data in validation_dataloader:
            logger.debug(f"Validating {validation_data=} on rank={parallel_state.rank}.")

            validation_data = validation_data[0]
            validation_artifacts = self.model_specification.validation(
                pipeline=pipeline,
                generator=torch.Generator(device=parallel_state.device).manual_seed(
                    self.args.seed if self.args.seed is not None else 0
                ),
                **validation_data,
            )

            PROMPT = validation_data["prompt"]
            IMAGE = validation_data.get("image", None)
            VIDEO = validation_data.get("video", None)
            EXPORT_FPS = validation_data.get("export_fps", 30)

            # 2.1. If there are any initial images or videos, they will be logged to keep track of them as
            # conditioning for generation.
            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            artifacts = {
                "image": data.ImageArtifact(value=IMAGE),
                "video": data.VideoArtifact(value=VIDEO),
            }

            # 2.2. Track the artifacts generated from validation
            for i, validation_artifact in enumerate(validation_artifacts):
                if validation_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": validation_artifact})

            # 2.3. Save the artifacts to the output directory and create appropriate logging objects
            # TODO(aryan): Currently, we only support WandB so we've hardcoded it here. Needs to be revisited.
            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact))
                filename = "validation-" if not final_validation else "final-"
                filename += f"{step}-{parallel_state.rank}-{index}-{prompt_filename}.{artifact.file_extension}"
                output_filename = os.path.join(self.args.output_dir, filename)

                if parallel_state.is_main_process and artifact.file_extension == "mp4":
                    main_process_prompts_to_filenames[PROMPT] = filename

                if artifact.type == "image" and artifact.value is not None:
                    logger.debug(f"Saving image to {output_filename}")
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif artifact.type == "video" and artifact.value is not None:
                    logger.debug(f"Saving video to {output_filename}")
                    export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))

        parallel_state.wait_for_everyone()

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline

        utils.free_memory()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(parallel_state.device)

        # Gather artifacts from all processes. We also need to flatten them since each process returns a list of artifacts.
        all_artifacts = [None] * parallel_state.world_size
        torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_state.is_main_process:
            tracker_key = "final" if final_validation else "validation"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_state.log({tracker_key: artifact_log_dict}, step=step)

            if self.args.push_to_hub and final_validation:
                video_filenames = list(main_process_prompts_to_filenames.values())
                prompts = list(main_process_prompts_to_filenames.keys())
                utils.save_model_card(
                    args=self.args, repo_id=self.state.repo_id, videos=video_filenames, validation_prompts=prompts
                )

        parallel_state.wait_for_everyone()
        if not final_validation:
            self.transformer.train()

    def evaluate(self) -> None:
        raise NotImplementedError("Evaluation has not been implemented yet.")

    def _init_distributed(self) -> None:
        # TODO: Accelerate disables native_amp for MPS. Probably need to do the same with implementation.
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_type, device_module = utils.get_device_info()

        device = torch.device(f"{device_type}:{local_rank}")
        device_module.set_device(device)

        self.state.parallel_state = parallel.FinetrainersParallelState(
            world_size=world_size,
            pp_degree=self.args.pp_degree,
            dp_degree=self.args.dp_degree,
            dp_shards=self.args.dp_shards,
            cp_degree=self.args.cp_degree,
            tp_degree=self.args.tp_degree,
        )

        backend = "nccl"
        timeout = datetime.timedelta(seconds=self.args.init_timeout)

        torch.distributed.init_process_group(backend, timeout=timeout)

        world_mesh = self.state.parallel_state.get_mesh(device_type)

        if self.state.parallel_state.data_parallel_enabled:
            self.state.dp_mesh = world_mesh["dp"]
            self.state.dp_degree = self.state.dp_mesh.size()
            self.state.dp_rank = self.state.dp_mesh.get_local_rank()
        else:
            self.state.dp_degree = 1
            self.state.dp_rank = 0

        if self.state.parallel_state.pipeline_parallel_enabled:
            self.state.pp_mesh = world_mesh["pp"]

        if self.args.seed is not None:
            utils.enable_determinism(self.args.seed, world_mesh)

    def _init_logging(self) -> None:
        if torch.distributed.get_rank() == 0:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized FineTrainers")

        trackers = ["wandb"]
        experiment_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.parallel_state.initialize_trackers(
            trackers, experiment_name=experiment_name, config=self._get_training_info(), log_dir=self.args.logging_dir
        )

    def _init_directories_and_repositories(self) -> None:
        if self.state.parallel_state.is_main_process:
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

        # Peform any patches needed for training
        if len(self.args.layerwise_upcasting_modules) > 0:
            perform_peft_patches()

    def _init_non_model_conditions(self) -> None:
        if ProcessorType.CAPTION_TEXT_DROPOUT in self.state.condition_types:
            params = {"dropout_p": self.args.caption_dropout_p}
            self.caption_preprocessing_conditions.append(get_condition(ProcessorType.CAPTION_TEXT_DROPOUT, params))
            self.state.condition_types.remove(ProcessorType.CAPTION_TEXT_DROPOUT)

        if ProcessorType.CAPTION_EMBEDDING_DROPOUT in self.state.condition_types:
            params = {"dropout_p": self.args.caption_dropout_p}
            self.caption_postprocessing_conditions.append(
                get_condition(ProcessorType.CAPTION_EMBEDDING_DROPOUT, params)
            )
            self.state.condition_types.remove(ProcessorType.CAPTION_EMBEDDING_DROPOUT)

    def _move_components_to_device(self, components: Optional[List[torch.nn.Module]] = None) -> None:
        if components is None:
            components = [self.text_encoder, self.text_encoder_2, self.text_encoder_3, self.transformer, self.vae]
        components = utils.get_non_null_items(components)
        for component in components:
            component.to(self.state.parallel_state.device)

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
        utils.free_memory()
        utils.synchronize_device()

    def _get_and_prepare_pipeline_for_validation(self, final_validation: bool = False) -> DiffusionPipeline:
        parallel_state = self.state.parallel_state
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"]

        if not final_validation:
            module_names.remove("transformer")
            pipeline = self.model_specification.load_pipeline(
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                tokenizer_3=self.tokenizer_3,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                text_encoder_3=self.text_encoder_3,
                # TODO(aryan): handle unwrapping for compiled modules
                # transformer=utils.unwrap_model(accelerator, self.transformer),
                transformer=self.transformer,
                vae=self.vae,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=True,
            )
        else:
            self._delete_components()

            # Load the transformer weights from the final checkpoint if performing full-finetune
            transformer = None
            if self.args.training_type == "full-finetune":
                transformer = self.model_specification.load_diffusion_models()["transformer"]

            pipeline = self.model_specification.load_pipeline(
                transformer=transformer,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=False,
                device=parallel_state.device,
            )

            # Load the LoRA weights if performing LoRA finetuning
            if self.args.training_type == "lora":
                pipeline.load_lora_weights(self.args.output_dir)

        components = [getattr(pipeline, module_name, None) for module_name in module_names]
        self._move_components_to_device(components)
        return pipeline

    def _disable_grad_for_models(self, models: List[torch.nn.Module]):
        for model in models:
            if model is not None:
                model.requires_grad_(False)

    def _enable_grad_for_models(self, models: List[torch.nn.Module]):
        for model in models:
            if model is not None:
                model.requires_grad_(True)

    def _get_training_info(self) -> Dict[str, Any]:
        info = self.args.to_dict()

        # Removing flow matching arguments when not using flow-matching objective
        diffusion_args = info.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__ if self.scheduler is not None else ""
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args

        info.update({"diffusion_arguments": filtered_diffusion_args})
        return info
