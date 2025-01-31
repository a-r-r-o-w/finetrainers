import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import datasets.distributed
import diffusers
import torch
import torch.backends
import transformers
from accelerate.utils import gather_object
from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, load_image, load_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig
from tqdm import tqdm

import wandb

from . import data, optimizer, utils
from .args import Args, validate_args
from .hooks import apply_layerwise_upcasting
from .logging import logger
from .models import ModelSpecification, get_model_specifiction_cls
from .patches import perform_peft_patches
from .processors import Processor, ProcessorType, get_condition
from .state import ParallelState, State, TrainState
from .trackers import initialize_trackers


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

        # Trainer-specific conditions
        self.caption_preprocessing_conditions: List[Processor] = []
        self.caption_postprocessing_conditions: List[Processor] = []

        self.state.model_name = self.args.model_name
        self.state.condition_types = self.args.conditions

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()
        self._init_config_options()
        self._init_non_model_conditions()

        # Peform any patches needed for training
        if len(self.args.layerwise_upcasting_modules) > 0:
            perform_peft_patches()
        # TODO(aryan): handle text encoders
        # if any(["text_encoder" in component_name for component_name in self.args.layerwise_upcasting_modules]):
        #     perform_text_encoder_patches()

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
        parallel_state = self.state.parallel
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

        if self.state.parallel.pipeline_parallel_enabled:
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

    def prepare_optimizer(self) -> None:
        logger.info("Initializing trainable parameters")

        if self.args.precompute_conditions:
            diffusion_components = self.model_specification.load_diffusion_models()
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

        # ============ TODO(aryan): cleanup

        logger.info("Initializing optimizer and lr scheduler")

        self.state.train_state = TrainState()

        # Make sure the trainable params are in float32
        if self.args.training_type == "lora":
            # TODO(aryan): handle lora parameters since optimizer expects nn.Module
            raise NotImplementedError
            cast_training_params([self.transformer], dtype=torch.float32)

        model_parts = [self.transformer]
        self.state.num_trainable_parameters = sum(
            p.numel() for m in model_parts for p in m.parameters() if p.requires_grad
        )

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
        if self.state.parallel.context_parallel_enabled:
            raise NotImplementedError(
                "Context parallelism is not supported yet. This will be supported in the future."
            )

        world_mesh = self.state.parallel.get_mesh()

        if self.state.parallel.data_sharding_enabled:
            if self.state.parallel.data_replication_enabled:
                logger.info("Applying HSDP to the model")
            else:
                logger.info("Applying FSDP to the model")

            # Apply FSDP or HSDP
            if self.state.parallel.data_replication_enabled or self.state.parallel.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            param_dtype = torch.float32 if self.args.training_type == "lora" else self.args.transformer_dtype

            utils.apply_fsdp(
                model=self.transformer,
                dp_mesh=world_mesh[dp_mesh_names],
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
                pp_enabled=self.state.parallel.pipeline_parallel_enabled,
                cpu_offload=False,  # TODO(aryan): needs to be tested and allowed for enabling later
            )
        elif self.state.parallel.data_replication_enabled:
            logger.info("Applying DDP to the model")

            if world_mesh.ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")

            utils.apply_ddp(model=self.transformer, dp_mesh=world_mesh)

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

        global_batch_size = self.args.batch_size * self.state.parallel.world_size
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-device batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        train_state = self.state.train_state
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

        parallel_state = self.state.parallel
        device = parallel_state.device

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
            logger.debug(f"Starting training step ({train_state.step}/{self.args.train_steps})")
            self.optimizer.zero_grad()

            batch = next(data_iterator)
            batch_size = len(batch["caption"])

            train_state.step += 1
            # TODO(aryan): this is not correct. We need to handle PP and DP properly
            train_state.observed_data_samples += batch_size * parallel_state._world_size

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
                    utils.dist_mean(loss, dp_cp_mesh),
                    utils.dist_max(loss, dp_cp_mesh),
                )
            else:
                global_avg_loss = global_max_loss = loss.detach().item()

            logs["global_avg_loss"] = global_avg_loss
            logs["global_max_loss"] = global_max_loss
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            self.tracker.log(logs, step=train_state.step)

            train_state.log_steps.append(train_state.step)
            train_state.global_avg_losses.append(global_avg_loss)
            train_state.global_max_losses.append(global_max_loss)

            # TODO(aryan): handle validation and checkpointing

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
        # self.validate(step=train_state.step, final_validation=True)

        if parallel_state.is_main_process:
            if self.args.push_to_hub:
                upload_folder(
                    repo_id=self.state.repo_id, folder_path=self.args.output_dir, ignore_patterns=["checkpoint-*"]
                )

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        self.tracker.finish()
        parallel_state.destroy()

    def validate(self, step: int, final_validation: bool = False) -> None:
        logger.info("Starting validation")

        accelerator = self.state.accelerator
        num_validation_samples = len(self.args.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.transformer.eval()

        memory_statistics = utils.get_memory_statistics()
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
            validation_artifacts = self.model_specification.validation(
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
                # TODO: support passing `fps` for supported pipelines
            )

            prompt_filename = utils.string_to_filename(prompt)[:25]
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
                utils.save_model_card(
                    args=self.args,
                    repo_id=self.state.repo_id,
                    videos=video_filenames,
                    validation_prompts=prompts,
                )

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline

        accelerator.wait_for_everyone()

        utils.free_memory()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

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

        self.state.parallel = ParallelState(
            world_size=world_size,
            pipeline_parallel_degree=self.args.pp_degree,
            data_parallel_degree=self.args.dp_degree,
            data_parallel_shards=self.args.dp_shards,
            context_parallel_degree=self.args.cp_degree,
            tensor_parallel_degree=self.args.tp_degree,
        )

        backend = "nccl"
        timeout = datetime.timedelta(seconds=self.args.init_timeout)

        torch.distributed.init_process_group(backend, timeout=timeout)

        world_mesh = self.state.parallel.get_mesh(device_type)

        if self.state.parallel.data_parallel_enabled:
            self.state.dp_mesh = world_mesh["dp"]
            self.state.dp_degree = self.state.dp_mesh.size()
            self.state.dp_rank = self.state.dp_mesh.get_local_rank()
        else:
            self.state.dp_degree = 1
            self.state.dp_rank = 0

        if self.state.parallel.pipeline_parallel_enabled:
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
        self.tracker = initialize_trackers(
            trackers, experiment_name, config=self._get_training_info(), log_dir=self.args.logging_dir
        )

    def _init_directories_and_repositories(self) -> None:
        if self.state.parallel.is_main_process:
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

    def _move_components_to_device(self) -> None:
        components = [self.text_encoder, self.text_encoder_2, self.text_encoder_3, self.transformer, self.vae]
        for component in components:
            if component is not None:
                component.to(self.state.parallel.device)

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
        parallel_state = self.state.parallel

        if not final_validation:
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
                device=parallel_state.device,
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

        # LoRA/non-LoRA stuff
        if training_type == "full-finetune":
            filtered_training_args = {
                k: v for k, v in training_args.items() if k not in {"rank", "lora_alpha", "target_modules"}
            }
        else:
            filtered_training_args = training_args

        # Diffusion/flow stuff
        diffusion_args = args.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args

        # Rest of the stuff
        updated_training_info = args.copy()
        updated_training_info["training_arguments"] = filtered_training_args
        updated_training_info["diffusion_arguments"] = filtered_diffusion_args
        return updated_training_info
