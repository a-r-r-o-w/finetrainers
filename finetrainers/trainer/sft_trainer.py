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

from .. import data, logging, optimizer, parallel, patches, utils
from ..args import Args, validate_args
from ..hooks import apply_layerwise_upcasting
from ..models import ModelSpecification
from ..processors import Processor, ProcessorType, get_condition
from ..state import State, TrainState


logger = logging.get_logger()


class SFTTrainer:
    def __init__(self, args: Args, model_specification: ModelSpecification) -> None:
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
        self._init_config_options()
        self._init_non_model_conditions()

        # Perform any patches that might be necessary for training to work as expected
        patches.perform_patches_for_training(self.args, self.state.parallel_backend)

        self.model_specification = model_specification

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

        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )

    def prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        parallel_backend = self.state.parallel_backend

        if self.args.precompute_conditions:
            diffusion_components = self.model_specification.load_diffusion_models()
            self._set_components(diffusion_components)

        components = [self.text_encoder, self.text_encoder_2, self.text_encoder_3, self.vae]
        utils.set_requires_grad(components, False)

        if self.args.training_type == "full-finetune":
            logger.info("Finetuning transformer with no additional parameters")
            utils.set_requires_grad([self.transformer], True)
        else:
            logger.info("Finetuning transformer with PEFT parameters")
            utils.set_requires_grad([self.transformer], False)

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

        # # TODO(aryan): it might be nice to add some assertions here to make sure that lora parameters are still in fp32
        # # even if layerwise upcasting. Would be nice to have a test as well
        # self.register_saving_loading_hooks(transformer_lora_config)

        # Make sure the trainable params are in float32 if data sharding is not enabled. For FSDP, we need all
        # parameters to be of the same dtype.
        if self.args.training_type == "lora":
            casting_dtype = (
                torch.float32 if not parallel_backend.data_sharding_enabled else self.args.transformer_dtype
            )
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
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
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
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
            # TODO(aryan): handle last_epoch
        )

    def prepare_for_training(self) -> None:
        # 1. Apply parallelism
        parallel_backend = self.state.parallel_backend
        world_mesh = parallel_backend.get_mesh()
        model_specification = self.model_specification

        if parallel_backend.context_parallel_enabled:
            raise NotImplementedError(
                "Context parallelism is not supported yet. This will be supported in the future."
            )

        if parallel_backend.tensor_parallel_enabled:
            # TODO(aryan): handle fp8 from TorchAO here
            model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh()["tp"],
                transformer=self.transformer,
            )

        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            # TODO(aryan): support other checkpointing types
            utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")

        # Enable DDP, FSDP or HSDP
        if parallel_backend.data_sharding_enabled:
            # TODO(aryan): remove this when supported
            if self.args.parallel_backend == "accelerate":
                raise NotImplementedError("Data sharding is not supported with Accelerate yet.")

            if parallel_backend.data_replication_enabled:
                logger.info("Applying HSDP to the model")
            else:
                logger.info("Applying FSDP to the model")

            # Apply FSDP or HSDP
            if parallel_backend.data_replication_enabled or parallel_backend.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            parallel.apply_fsdp2_ptd(
                model=self.transformer,
                dp_mesh=world_mesh[dp_mesh_names],
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,  # TODO(aryan): needs to be tested and allowed for enabling later
            )
        elif parallel_backend.data_replication_enabled:
            logger.info("Applying DDP to the model")

            if world_mesh.ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")

            parallel_backend.apply_ddp(self.transformer, world_mesh)

        self._move_components_to_device()

        # 2. Prepare optimizer and lr scheduler
        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(self.optimizer, self.lr_scheduler)

        # 3. Initialize trackers, directories and repositories
        self._init_logging()
        self._init_trackers()
        self._init_directories_and_repositories()

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        # TODO(aryan): allow configurability
        dataset = data.initialize_dataset(self.args.data_root, dataset_type="video", streaming=True, infinite=True)

        # TODO(aryan): support batch size > 1
        self.dataset, self.dataloader = self.state.parallel_backend.prepare_dataset(
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.pin_memory,
        )

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
        global_batch_size = self.args.batch_size * self.state.parallel_backend.world_size
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-replica batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        train_state = self.state.train_state
        parallel_backend = self.state.parallel_backend
        device = parallel_backend.device

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
            disable=not parallel_backend.is_local_main_process,
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
            train_state.observed_data_samples += batch_size * parallel_backend._world_size

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

            if parallel_backend.pipeline_parallel_enabled:
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
            grad_norm = utils.torch_utils._clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                self.args.max_grad_norm,
                foreach=True,
                pp_mesh=parallel_backend.get_mesh()["pp"] if parallel_backend.pipeline_parallel_enabled else None,
            )

            self.optimizer.step()
            self.lr_scheduler.step()

            logs = {}
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.detach().item()
            if (
                parallel_backend.data_replication_enabled
                or parallel_backend.data_sharding_enabled
                or parallel_backend.context_parallel_enabled
            ):
                loss = loss.detach()
                dp_cp_mesh = parallel_backend.get_mesh()["dp_cp"]
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
                parallel_backend.log(logs, step=train_state.step)

            train_state.log_steps.append(train_state.step)
            train_state.global_avg_losses.append(global_avg_loss)
            train_state.global_max_losses.append(global_max_loss)

            # TODO(aryan): handle checkpointing
            if train_state.step % self.args.validation_steps == 0:
                self.validate(step=train_state.step, final_validation=False)

        parallel_backend.wait_for_everyone()

        if parallel_backend.is_main_process:
            # TODO(aryan): handle compiled models by unwrapping and exporting model
            # transformer = utils.unwrap_model(accelerator, self.transformer)

            # if self.args.training_type == "lora":
            #     transformer_lora_layers = get_peft_model_state_dict(transformer)
            #     self.model_specification.save_lora_weights(self.args.output_dir, transformer_lora_layers)
            # else:
            #     self.model_specification.save_model(self.args.output_dir, transformer=transformer)
            pass

        parallel_backend.wait_for_everyone()
        self.validate(step=train_state.step, final_validation=True)

        if parallel_backend.is_main_process:
            if self.args.push_to_hub:
                upload_folder(
                    repo_id=self.state.repo_id, folder_path=self.args.output_dir, ignore_patterns=["checkpoint-*"]
                )

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        parallel_backend.destroy()

    def validate(self, step: int, final_validation: bool = False) -> None:
        logger.info("Starting validation")

        # 1. Load validation dataset
        parallel_backend = self.state.parallel_backend
        world_mesh = parallel_backend.get_mesh()

        if "dp" in world_mesh.mesh_dim_names:
            dp_mesh = world_mesh["dp"]
            local_rank, dp_world_size = dp_mesh.local_rank, dp_mesh.size()
        else:
            local_rank, dp_world_size = 0, 1

        dataset = data.ValidationDataset(self.args.validation_dataset_file)
        dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, local_rank, dp_world_size)
        validation_dataloader = data.DPDataLoader(
            local_rank,
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items,
        )
        data_iterator = iter(validation_dataloader)
        main_process_prompts_to_filenames = {}  # Used to save model card
        all_processes_artifacts = []  # Used to gather artifacts from all processes

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)
        pipeline = self._init_pipeline(final_validation=final_validation)

        # 2. Run validation
        # TODO(aryan): when running validation with FSDP, if the number of data points is not divisible by dp_shards, we
        # will hang indefinitely. Either pad the dataset or raise an error early on during initialization if the dataset
        # size is not divisible by dp_shards.
        self.transformer.eval()
        while True:
            validation_data = next(data_iterator, None)
            if validation_data is None:
                break

            logger.debug(
                f"Validating {validation_data=} on rank={parallel_backend.rank}.", local_main_process_only=False
            )

            validation_data = validation_data[0]
            validation_artifacts = self.model_specification.validation(
                pipeline=pipeline, generator=generator, **validation_data
            )

            PROMPT = validation_data["prompt"]
            IMAGE = validation_data.get("image", None)
            VIDEO = validation_data.get("video", None)
            EXPORT_FPS = validation_data.get("export_fps", 30)

            # 2.1. If there are any initial images or videos, they will be logged to keep track of them as
            # conditioning for generation.
            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            artifacts = {
                "input_image": data.ImageArtifact(value=IMAGE),
                "input_video": data.VideoArtifact(value=VIDEO),
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
                filename += f"{step}-{parallel_backend.rank}-{index}-{prompt_filename}.{artifact.file_extension}"
                output_filename = os.path.join(self.args.output_dir, filename)

                if parallel_backend.is_main_process and artifact.file_extension == "mp4":
                    main_process_prompts_to_filenames[PROMPT] = filename

                if artifact.type == "image" and artifact.value is not None:
                    logger.debug(
                        f"Saving image from rank={parallel_backend.rank} to {output_filename}",
                        local_main_process_only=False,
                    )
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif artifact.type == "video" and artifact.value is not None:
                    logger.debug(
                        f"Saving video from rank={parallel_backend.rank} to {output_filename}",
                        local_main_process_only=False,
                    )
                    export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))

        # 3. Cleanup & log artifacts
        parallel_backend.wait_for_everyone()

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline

        utils.free_memory()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(parallel_backend.device)

        # Gather artifacts from all processes. We also need to flatten them since each process returns a list of artifacts.
        # TODO(aryan): probably should only all gather from dp mesh process group
        all_artifacts = [None] * parallel_backend.world_size
        torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_backend.is_main_process:
            tracker_key = "final" if final_validation else "validation"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_backend.log({tracker_key: artifact_log_dict}, step=step)

            if self.args.push_to_hub and final_validation:
                video_filenames = list(main_process_prompts_to_filenames.values())
                prompts = list(main_process_prompts_to_filenames.keys())
                utils.save_model_card(
                    args=self.args, repo_id=self.state.repo_id, videos=video_filenames, validation_prompts=prompts
                )

        parallel_backend.wait_for_everyone()
        if not final_validation:
            self.transformer.train()

    def evaluate(self) -> None:
        raise NotImplementedError("Evaluation has not been implemented yet.")

    def _init_distributed(self) -> None:
        # TODO: Accelerate disables native_amp for MPS. Probably need to do the same with implementation.
        world_size = int(os.environ["WORLD_SIZE"])

        # TODO(aryan): handle other backends
        backend_cls: parallel.ParallelBackendType = parallel.get_parallel_backend_cls(self.args.parallel_backend)
        self.state.parallel_backend = backend_cls(
            world_size=world_size,
            pp_degree=self.args.pp_degree,
            dp_degree=self.args.dp_degree,
            dp_shards=self.args.dp_shards,
            cp_degree=self.args.cp_degree,
            tp_degree=self.args.tp_degree,
            backend="nccl",
            timeout=self.args.init_timeout,
            logging_dir=self.args.logging_dir,
            output_dir=self.args.output_dir,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        if self.args.seed is not None:
            world_mesh = self.state.parallel_backend.get_mesh()
            utils.enable_determinism(self.args.seed, world_mesh)

    def _init_logging(self) -> None:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        logging._set_parallel_backend(self.state.parallel_backend)
        logger.info("Initialized FineTrainers")

    def _init_trackers(self) -> None:
        # TODO(aryan): handle multiple trackers
        trackers = ["wandb"]
        experiment_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.parallel_backend.initialize_trackers(
            trackers, experiment_name=experiment_name, config=self._get_training_info(), log_dir=self.args.logging_dir
        )

    def _init_directories_and_repositories(self) -> None:
        if self.state.parallel_backend.is_main_process:
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

    def _move_components_to_device(self, components: Optional[List[torch.nn.Module]] = None) -> None:
        if components is None:
            components = [self.text_encoder, self.text_encoder_2, self.text_encoder_3, self.transformer, self.vae]
        components = utils.get_non_null_items(components)
        for component in components:
            component.to(self.state.parallel_backend.device)

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

    def _init_pipeline(self, final_validation: bool = False) -> DiffusionPipeline:
        parallel_backend = self.state.parallel_backend
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
            # TODO(aryan): this branch does not work yet, needs to be implemented
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
                device=parallel_backend.device,
            )

            # Load the LoRA weights if performing LoRA finetuning
            if self.args.training_type == "lora":
                pipeline.load_lora_weights(self.args.output_dir)

        components = [getattr(pipeline, module_name, None) for module_name in module_names]
        self._move_components_to_device(components)
        return pipeline

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
