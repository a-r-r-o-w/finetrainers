import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets.distributed
import torch
import wandb
from diffusers.hooks import HookRegistry, ModelHook
from diffusers.utils import export_to_video

from finetrainers import data, get_logger, logging, parallel, patches, utils
from finetrainers.args import AttentionProviderInference
from finetrainers.config import ModelType
from finetrainers.models import ModelSpecification, attention_provider
from finetrainers.models.cogvideox import CogVideoXModelSpecification
from finetrainers.models.cogview4 import CogView4ModelSpecification
from finetrainers.models.flux import FluxModelSpecification
from finetrainers.models.wan import WanModelSpecification
from finetrainers.parallel import ParallelBackendEnum
from finetrainers.state import ParallelBackendType
from finetrainers.utils import ArgsConfigMixin


logger = get_logger()


def main():
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            f'Failed to set multiprocessing start method to "fork". This can lead to poor performance, high memory usage, or crashes. '
            f"See: https://pytorch.org/docs/stable/notes/multiprocessing.html\n"
            f"Error: {e}"
        )

    try:
        args = BaseArgs()
        args.parse_args()

        model_specification_cls = get_model_specifiction_cls(args.model_name, args.inference_type)
        model_specification = model_specification_cls(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_2_id=args.tokenizer_2_id,
            tokenizer_3_id=args.tokenizer_3_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            text_encoder_3_id=args.text_encoder_3_id,
            transformer_id=args.transformer_id,
            vae_id=args.vae_id,
            text_encoder_dtype=args.text_encoder_dtype,
            text_encoder_2_dtype=args.text_encoder_2_dtype,
            text_encoder_3_dtype=args.text_encoder_3_dtype,
            transformer_dtype=args.transformer_dtype,
            vae_dtype=args.vae_dtype,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )

        inferencer = Inference(args, model_specification)
        inferencer.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error(traceback.format_exc())


class InferenceType(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"


# We do a union because every ArgsConfigMixin registered to BaseArgs can be looked up using the `__getattribute__` override
BaseArgsType = Union[
    "BaseArgs", "ParallelArgs", "ModelArgs", "InferenceArgs", "AttentionProviderArgs", "TorchConfigArgs"
]

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


SUPPORTED_MODEL_CONFIGS = {
    ModelType.COGVIDEOX: {
        InferenceType.TEXT_TO_VIDEO: CogVideoXModelSpecification,
    },
    ModelType.COGVIEW4: {
        InferenceType.TEXT_TO_IMAGE: CogView4ModelSpecification,
    },
    ModelType.FLUX: {
        InferenceType.TEXT_TO_IMAGE: FluxModelSpecification,
    },
    ModelType.WAN: {
        InferenceType.TEXT_TO_VIDEO: WanModelSpecification,
        InferenceType.IMAGE_TO_VIDEO: WanModelSpecification,
    },
}


def get_model_specifiction_cls(model_name: str, inference_type: InferenceType) -> ModelSpecification:
    """
    Get the model specification class for the given model name and inference type.
    """
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if inference_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Inference type {inference_type} not supported for model {model_name}. Supported inference types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][inference_type]


@dataclass
class State:
    # Parallel state
    parallel_backend: ParallelBackendType = None

    # Training state
    generator: torch.Generator = None


class Inference:
    def __init__(self, args: BaseArgsType, model_specification: ModelSpecification):
        self.args = args
        self.model_specification = model_specification
        self.state = State()

        self.pipeline = None
        self.dataset = None
        self.dataloader = None

        self._init_distributed()
        self._init_config_options()

        patches.perform_patches_for_inference(args, self.state.parallel_backend)

    def run(self) -> None:
        try:
            self._prepare_pipeline()
            self._prepare_distributed()
            self._prepare_dataset()
            self._inference()
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def _prepare_pipeline(self) -> None:
        logger.info("Initializing pipeline")

        transformer = self.model_specification.load_diffusion_models()["transformer"]
        self.pipeline = self.model_specification.load_pipeline(
            transformer=transformer,
            enable_slicing=self.args.enable_slicing,
            enable_tiling=self.args.enable_tiling,
            enable_model_cpu_offload=False,  # TODO(aryan): handle model/sequential/group offloading
            training=False,
        )

    def _prepare_distributed(self) -> None:
        parallel_backend = self.state.parallel_backend
        cp_mesh = parallel_backend.get_mesh("cp") if parallel_backend.context_parallel_enabled else None

        if parallel_backend.context_parallel_enabled:
            cp_mesh = parallel_backend.get_mesh()["cp"]
            parallel_backend.apply_context_parallel(self.pipeline.transformer, cp_mesh)

        registry = HookRegistry.check_if_exists_or_initialize(self.pipeline.transformer)
        hook = AttentionProviderHook(
            self.args.attn_provider, cp_mesh, self.args.cp_rotate_method, self.args.cp_reduce_precision
        )
        registry.register_hook(hook, "attn_provider")

        self._maybe_torch_compile()

        self._init_logging()
        self._init_trackers()
        self._init_directories()

    def _prepare_dataset(self) -> None:
        logger.info("Preparing dataset for inference")
        parallel_backend = self.state.parallel_backend

        dp_mesh = None
        if parallel_backend.data_replication_enabled:
            dp_mesh = parallel_backend.get_mesh("dp_replicate")
        if dp_mesh is not None:
            local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        else:
            local_rank, dp_world_size = 0, 1

        dataset = data.ValidationDataset(self.args.dataset_file)
        dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, local_rank, dp_world_size)
        dataloader = data.DPDataLoader(
            local_rank,
            dataset,
            batch_size=1,
            num_workers=0,  # TODO(aryan): handle dataloader_num_workers
            collate_fn=lambda items: items,
        )

        self.dataset = dataset
        self.dataloader = dataloader

    def _inference(self) -> None:
        parallel_backend = self.state.parallel_backend
        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)

        if parallel_backend._dp_degree > 1:
            dp_mesh = parallel_backend.get_mesh("dp")
            dp_local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        else:
            dp_mesh = None
            dp_local_rank, dp_world_size = parallel_backend.local_rank, 1

        self.pipeline.to(parallel_backend.device)

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before inference start: {json.dumps(memory_statistics, indent=4)}")

        data_iterator = iter(self.dataloader)
        main_process_prompts_to_filenames = {}  # Used to save model card
        all_processes_artifacts = []  # Used to gather artifacts from all processes

        while True:
            inference_data = next(data_iterator, None)
            if inference_data is None:
                break

            inference_data = inference_data[0]
            with torch.inference_mode():
                inference_artifacts = self.model_specification.validation(
                    pipeline=self.pipeline, generator=generator, **inference_data
                )

            if dp_local_rank != 0:
                continue

            PROMPT = inference_data["prompt"]
            IMAGE = inference_data.get("image", None)
            VIDEO = inference_data.get("video", None)
            EXPORT_FPS = inference_data.get("export_fps", 30)

            # 2.1. If there are any initial images or videos, they will be logged to keep track of them as
            # conditioning for generation.
            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            artifacts = {
                "input_image": data.ImageArtifact(value=IMAGE),
                "input_video": data.VideoArtifact(value=VIDEO),
            }

            # 2.2. Track the artifacts generated from inference
            for i, inference_artifact in enumerate(inference_artifacts):
                if inference_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": inference_artifact})

            # 2.3. Save the artifacts to the output directory and create appropriate logging objects
            # TODO(aryan): Currently, we only support WandB so we've hardcoded it here. Needs to be revisited.
            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact))
                if artifact.value is None:
                    continue

                time_, rank, ext = int(time.time()), parallel_backend.rank, artifact.file_extension
                filename = f"inference-{rank}-{index}-{prompt_filename}-{time_}.{ext}"
                output_filename = os.path.join(self.args.output_dir, filename)

                if parallel_backend.is_main_process and ext in ["mp4", "jpg", "jpeg", "png"]:
                    main_process_prompts_to_filenames[PROMPT] = filename

                if isinstance(artifact, data.ImageArtifact):
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif isinstance(artifact, data.VideoArtifact):
                    export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))

        # 3. Cleanup & log artifacts
        parallel_backend.wait_for_everyone()

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after inference end: {json.dumps(memory_statistics, indent=4)}")

        # Gather artifacts from all processes. We also need to flatten them since each process returns a list of artifacts.
        all_artifacts = [None] * dp_world_size
        if dp_world_size > 1:
            torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        else:
            all_artifacts = [all_processes_artifacts]
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_backend.is_main_process:
            tracker_key = "inference"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts
            parallel_backend.log({tracker_key: artifact_log_dict}, step=0)

        parallel_backend.wait_for_everyone()

    def _init_distributed(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

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
        )

        if self.args.seed is not None:
            self.state.parallel_backend.enable_determinism(self.args.seed)

    def _init_logging(self) -> None:
        logging._set_parallel_backend(self.state.parallel_backend)
        logging.set_dependency_log_level(self.args.verbose, self.state.parallel_backend.is_local_main_process)
        logger.info("Initialized Finetrainers")

    def _init_trackers(self) -> None:
        # TODO(aryan): handle multiple trackers
        trackers = [self.args.report_to]
        experiment_name = self.args.tracker_name or "finetrainers-inference"
        self.state.parallel_backend.initialize_trackers(
            trackers, experiment_name=experiment_name, config=self.args.to_dict(), log_dir=self.args.logging_dir
        )

    def _init_directories(self) -> None:
        if self.state.parallel_backend.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_config_options(self) -> None:
        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision(self.args.float32_matmul_precision)

    def _maybe_torch_compile(self):
        for model_name, compile_scope in zip(self.args.compile_modules, self.args.compile_scopes):
            model = getattr(self.pipeline, model_name, None)
            if model is not None:
                logger.info(f"Applying torch.compile to '{model_name}' with scope '{compile_scope}'.")
                compiled_model = utils.apply_compile(model, compile_scope)
                setattr(self.pipeline, model_name, compiled_model)


class AttentionProviderHook(ModelHook):
    def __init__(
        self,
        provider: str,
        mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
        rotate_method: str = "allgather",
        reduce_precision: bool = False,
    ):
        super().__init__()
        self.provider = provider
        self.mesh = mesh
        self.rotate_method = rotate_method
        self.convert_to_fp32 = not reduce_precision

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        with attention_provider(
            self.provider, mesh=self.mesh, convert_to_fp32=self.convert_to_fp32, rotate_method=self.rotate_method
        ):
            return self.fn_ref.original_forward(*args, **kwargs)


class ParallelArgs(ArgsConfigMixin):
    """
    Args:
        parallel_backend (`str`, defaults to "accelerate"):
            The parallel backend to use for inference. Choose between ['accelerate', 'ptd'].
        pp_degree (`int`, defaults to 1):
            The degree of pipeline parallelism.
        dp_degree (`int`, defaults to 1):
            The degree of data parallelism (number of model replicas).
        dp_shards (`int`, defaults to 1):
            The number of data parallel shards (number of model partitions).
        cp_degree (`int`, defaults to 1):
            The degree of context parallelism.
    """

    parallel_backend: ParallelBackendEnum = ParallelBackendEnum.ACCELERATE
    pp_degree: int = 1
    dp_degree: int = 1
    dp_shards: int = 1
    cp_degree: int = 1
    tp_degree: int = 1
    cp_rotate_method: str = "allgather"
    cp_reduce_precision: bool = False

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--parallel_backend", type=str, default="accelerate", choices=["accelerate", "ptd"])
        parser.add_argument("--pp_degree", type=int, default=1)
        parser.add_argument("--dp_degree", type=int, default=1)
        parser.add_argument("--dp_shards", type=int, default=1)
        parser.add_argument("--cp_degree", type=int, default=1)
        parser.add_argument("--tp_degree", type=int, default=1)
        parser.add_argument("--cp_rotate_method", type=str, default="allgather", choices=["allgather", "alltoall"])
        parser.add_argument("--cp_reduce_precision", action="store_true")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.parallel_backend = argparse_args.parallel_backend
        mapped_args.pp_degree = argparse_args.pp_degree
        mapped_args.dp_degree = argparse_args.dp_degree
        mapped_args.dp_shards = argparse_args.dp_shards
        mapped_args.cp_degree = argparse_args.cp_degree
        mapped_args.tp_degree = argparse_args.tp_degree
        mapped_args.cp_rotate_method = argparse_args.cp_rotate_method
        mapped_args.cp_reduce_precision = argparse_args.cp_reduce_precision

    def validate_args(self, args: "BaseArgs"):
        if args.parallel_backend != "ptd":
            raise ValueError("Only 'ptd' parallel backend is supported for now.")
        if any(x > 1 for x in [args.pp_degree, args.dp_degree, args.dp_shards, args.tp_degree]):
            raise ValueError("Parallel degrees must be 1 except for `cp_degree` for now.")


class ModelArgs(ArgsConfigMixin):
    """
    Args:
        model_name (`str`):
            Name of model to train.
        pretrained_model_name_or_path (`str`):
            Path to pretrained model or model identifier from https://huggingface.co/models. The model should be
            loadable based on specified `model_name`.
        revision (`str`, defaults to `None`):
            If provided, the model will be loaded from a specific branch of the model repository.
        variant (`str`, defaults to `None`):
            Variant of model weights to use. Some models provide weight variants, such as `fp16`, to reduce disk
            storage requirements.
        cache_dir (`str`, defaults to `None`):
            The directory where the downloaded models and datasets will be stored, or loaded from.
        tokenizer_id (`str`, defaults to `None`):
            Identifier for the tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
        tokenizer_2_id (`str`, defaults to `None`):
            Identifier for the second tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
        tokenizer_3_id (`str`, defaults to `None`):
            Identifier for the third tokenizer model. This is useful when using a different tokenizer than the default from `pretrained_model_name_or_path`.
        text_encoder_id (`str`, defaults to `None`):
            Identifier for the text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
        text_encoder_2_id (`str`, defaults to `None`):
            Identifier for the second text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
        text_encoder_3_id (`str`, defaults to `None`):
            Identifier for the third text encoder model. This is useful when using a different text encoder than the default from `pretrained_model_name_or_path`.
        transformer_id (`str`, defaults to `None`):
            Identifier for the transformer model. This is useful when using a different transformer model than the default from `pretrained_model_name_or_path`.
        vae_id (`str`, defaults to `None`):
            Identifier for the VAE model. This is useful when using a different VAE model than the default from `pretrained_model_name_or_path`.
        text_encoder_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type for the text encoder when generating text embeddings.
        text_encoder_2_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type for the text encoder 2 when generating text embeddings.
        text_encoder_3_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type for the text encoder 3 when generating text embeddings.
        transformer_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type for the transformer model.
        vae_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Data type for the VAE model.
        layerwise_upcasting_modules (`List[str]`, defaults to `[]`):
            Modules that should have fp8 storage weights but higher precision computation. Choose between ['transformer'].
        layerwise_upcasting_storage_dtype (`torch.dtype`, defaults to `float8_e4m3fn`):
            Data type for the layerwise upcasting storage. Choose between ['float8_e4m3fn', 'float8_e5m2'].
        layerwise_upcasting_skip_modules_pattern (`List[str]`, defaults to `["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$", "^proj_out$", "norm"]`):
            Modules to skip for layerwise upcasting. Layers such as normalization and modulation, when casted to fp8 precision
            naively (as done in layerwise upcasting), can lead to poorer training and inference quality. We skip these layers
            by default, and recommend adding more layers to the default list based on the model architecture.
        enable_slicing (`bool`, defaults to `False`):
            Whether to enable VAE slicing.
        enable_tiling (`bool`, defaults to `False`):
            Whether to enable VAE tiling.
    """

    model_name: str = None
    pretrained_model_name_or_path: str = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    cache_dir: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_2_id: Optional[str] = None
    tokenizer_3_id: Optional[str] = None
    text_encoder_id: Optional[str] = None
    text_encoder_2_id: Optional[str] = None
    text_encoder_3_id: Optional[str] = None
    transformer_id: Optional[str] = None
    vae_id: Optional[str] = None
    text_encoder_dtype: torch.dtype = torch.bfloat16
    text_encoder_2_dtype: torch.dtype = torch.bfloat16
    text_encoder_3_dtype: torch.dtype = torch.bfloat16
    transformer_dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype = torch.bfloat16
    layerwise_upcasting_modules: List[str] = []
    layerwise_upcasting_storage_dtype: torch.dtype = torch.float8_e4m3fn
    # fmt: off
    layerwise_upcasting_skip_modules_pattern: List[str] = ["patch_embed", "pos_embed", "x_embedder", "context_embedder", "time_embed", "^proj_in$", "^proj_out$", "norm"]
    # fmt: on
    enable_slicing: bool = False
    enable_tiling: bool = False

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model_name", type=str, required=True, choices=[x.value for x in ModelType.__members__.values()]
        )
        parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
        parser.add_argument("--revision", type=str, default=None, required=False)
        parser.add_argument("--variant", type=str, default=None)
        parser.add_argument("--cache_dir", type=str, default=None)
        parser.add_argument("--tokenizer_id", type=str, default=None)
        parser.add_argument("--tokenizer_2_id", type=str, default=None)
        parser.add_argument("--tokenizer_3_id", type=str, default=None)
        parser.add_argument("--text_encoder_id", type=str, default=None)
        parser.add_argument("--text_encoder_2_id", type=str, default=None)
        parser.add_argument("--text_encoder_3_id", type=str, default=None)
        parser.add_argument("--transformer_id", type=str, default=None)
        parser.add_argument("--vae_id", type=str, default=None)
        parser.add_argument("--text_encoder_dtype", type=str, default="bf16")
        parser.add_argument("--text_encoder_2_dtype", type=str, default="bf16")
        parser.add_argument("--text_encoder_3_dtype", type=str, default="bf16")
        parser.add_argument("--transformer_dtype", type=str, default="bf16")
        parser.add_argument("--vae_dtype", type=str, default="bf16")
        parser.add_argument("--layerwise_upcasting_modules", type=str, default=[], nargs="+", choices=["transformer"])
        parser.add_argument(
            "--layerwise_upcasting_storage_dtype",
            type=str,
            default="float8_e4m3fn",
            choices=["float8_e4m3fn", "float8_e5m2"],
        )
        parser.add_argument(
            "--layerwise_upcasting_skip_modules_pattern",
            type=str,
            default=["patch_embed", "pos_embed", "x_embedder", "context_embedder", "^proj_in$", "^proj_out$", "norm"],
            nargs="+",
        )
        parser.add_argument("--enable_slicing", action="store_true")
        parser.add_argument("--enable_tiling", action="store_true")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.model_name = argparse_args.model_name
        mapped_args.pretrained_model_name_or_path = argparse_args.pretrained_model_name_or_path
        mapped_args.revision = argparse_args.revision
        mapped_args.variant = argparse_args.variant
        mapped_args.cache_dir = argparse_args.cache_dir
        mapped_args.tokenizer_id = argparse_args.tokenizer_id
        mapped_args.tokenizer_2_id = argparse_args.tokenizer_2_id
        mapped_args.tokenizer_3_id = argparse_args.tokenizer_3_id
        mapped_args.text_encoder_id = argparse_args.text_encoder_id
        mapped_args.text_encoder_2_id = argparse_args.text_encoder_2_id
        mapped_args.text_encoder_3_id = argparse_args.text_encoder_3_id
        mapped_args.transformer_id = argparse_args.transformer_id
        mapped_args.vae_id = argparse_args.vae_id
        mapped_args.text_encoder_dtype = _DTYPE_MAP[argparse_args.text_encoder_dtype]
        mapped_args.text_encoder_2_dtype = _DTYPE_MAP[argparse_args.text_encoder_2_dtype]
        mapped_args.text_encoder_3_dtype = _DTYPE_MAP[argparse_args.text_encoder_3_dtype]
        mapped_args.transformer_dtype = _DTYPE_MAP[argparse_args.transformer_dtype]
        mapped_args.vae_dtype = _DTYPE_MAP[argparse_args.vae_dtype]
        mapped_args.layerwise_upcasting_modules = argparse_args.layerwise_upcasting_modules
        mapped_args.layerwise_upcasting_storage_dtype = _DTYPE_MAP[argparse_args.layerwise_upcasting_storage_dtype]
        mapped_args.layerwise_upcasting_skip_modules_pattern = argparse_args.layerwise_upcasting_skip_modules_pattern
        mapped_args.enable_slicing = argparse_args.enable_slicing
        mapped_args.enable_tiling = argparse_args.enable_tiling

    def validate_args(self, args: "BaseArgs"):
        pass


class InferenceArgs(ArgsConfigMixin):
    """
    Args:
        inference_type (`str`):
            The type of inference to run. Choose between ['text_to_video'].
        dataset_file (`str`, defaults to `None`):
            Path to a CSV/JSON/PARQUET/ARROW file containing information for inference. The file must contain atleast the
            "caption" column. Other columns such as "image_path" and "video_path" can be provided too. If provided, "image_path"
            will be used to load a PIL.Image.Image and set the "image" key in the sample dictionary. Similarly, "video_path"
            will be used to load a List[PIL.Image.Image] and set the "video" key in the sample dictionary.
            The dataset file may contain other attributes such as:
                - "height" and "width" and "num_frames": Resolution
                - "num_inference_steps": Number of inference steps
                - "guidance_scale": Classifier-free Guidance Scale
                - ... (any number of additional attributes can be provided. The ModelSpecification::validate method will be
                invoked with the sample dictionary to validate the sample.)
    """

    inference_type: InferenceType = InferenceType.TEXT_TO_VIDEO
    dataset_file: str = None

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--inference_type",
            type=str,
            default=InferenceType.TEXT_TO_VIDEO.value,
            choices=[x.value for x in InferenceType.__members__.values()],
        )
        parser.add_argument("--dataset_file", type=str, required=True)

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.inference_type = InferenceType(argparse_args.inference_type)
        mapped_args.dataset_file = argparse_args.dataset_file

    def validate_args(self, args: "BaseArgs"):
        pass


class AttentionProviderArgs(ArgsConfigMixin):
    """
    Args:
        attn_provider (`str`, defaults to "native"):
            The attention provider to use for inference. Choose between ['flash', 'flash_varlen', 'flex', 'native', '_native_cudnn', '_native_efficient', '_native_flash', '_native_math', 'sage', 'sage_varlen', '_sage_qk_int8_pv_fp8_cuda', '_sage_qk_int8_pv_fp8_cuda_sm90', '_sage_qk_int8_pv_fp16_cuda', '_sage_qk_int8_pv_fp16_triton', 'xformers'].
    """

    attn_provider: AttentionProviderInference = "native"
    # attn_provider_specialized_modules: List[str] = []

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        # fmt: off
        parser.add_argument("--attn_provider", type=str, default="native", choices=["flash", "flash_varlen", "flex", "native", "_native_cudnn", "_native_efficient", "_native_flash", "_native_math", "sage", "sage_varlen", "_sage_qk_int8_pv_fp8_cuda", "_sage_qk_int8_pv_fp8_cuda_sm90", "_sage_qk_int8_pv_fp16_cuda", "_sage_qk_int8_pv_fp16_triton", "xformers"])
        # fmt: on

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.attn_provider = argparse_args.attn_provider

    def validate_args(self, args: "BaseArgs"):
        pass


class TorchConfigArgs(ArgsConfigMixin):
    """
    Args:
        compile_modules (`List[str]`, defaults to `[]`):
            Modules that should be regionally compiled with `torch.compile`.
        compile_scopes (`str`, defaults to `None`):
            The scope of compilation for each `--compile_modules`. Choose between ['regional', 'full']. Must have the same length as
            `--compile_modules`. If `None`, will default to `regional` for all modules.
        allow_tf32 (`bool`, defaults to `False`):
            Whether or not to allow the use of TF32 matmul on compatible hardware.
        float32_matmul_precision (`str`, defaults to `highest`):
            The precision to use for float32 matmul. Choose between ['highest', 'high', 'medium'].
    """

    compile_modules: List[str] = []
    compile_scopes: List[str] = None
    allow_tf32: bool = False
    float32_matmul_precision: str = "highest"

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--compile_modules", type=str, default=[], nargs="+")
        parser.add_argument("--compile_scopes", type=str, default=None, nargs="+")
        parser.add_argument("--allow_tf32", action="store_true")
        parser.add_argument(
            "--float32_matmul_precision",
            type=str,
            default="highest",
            choices=["highest", "high", "medium"],
            help="The precision to use for float32 matmul. Choose between ['highest', 'high', 'medium'].",
        )

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        compile_scopes = argparse_args.compile_scopes
        if len(argparse_args.compile_modules) > 0:
            if compile_scopes is None:
                compile_scopes = "regional"
            if isinstance(compile_scopes, list) and len(compile_scopes) == 1:
                compile_scopes = compile_scopes[0]
            if isinstance(compile_scopes, str):
                compile_scopes = [compile_scopes] * len(argparse_args.compile_modules)
        else:
            compile_scopes = []

        mapped_args.compile_modules = argparse_args.compile_modules
        mapped_args.compile_scopes = compile_scopes
        mapped_args.allow_tf32 = argparse_args.allow_tf32
        mapped_args.float32_matmul_precision = argparse_args.float32_matmul_precision

    def validate_args(self, args: "BaseArgs"):
        if len(args.compile_modules) > 0:
            assert len(args.compile_modules) == len(args.compile_scopes) and all(
                x in ["regional", "full"] for x in args.compile_scopes
            ), (
                "Compile modules and compile scopes must be of the same length and compile scopes must be either 'regional' or 'full'"
            )


class MiscellaneousArgs(ArgsConfigMixin):
    """
    Args:
        seed (`int`, defaults to `None`):
            Random seed for reproducibility under same initialization conditions.
        tracker_name (`str`, defaults to `finetrainers`):
            Name of the tracker/project to use for logging inference metrics.
        output_dir (`str`, defaults to `None`):
            The directory where the model checkpoints and logs will be stored.
        logging_dir (`str`, defaults to `logs`):
            The directory where the logs will be stored.
        logging_steps (`int`, defaults to `1`):
            Inference logs will be tracked every `logging_steps` steps.
        nccl_timeout (`int`, defaults to `1800`):
            Timeout for the NCCL communication.
        report_to (`str`, defaults to `wandb`):
            The name of the logger to use for logging inference metrics. Choose between ['wandb'].
        verbose (`int`, defaults to `1`):
            Whether or not to print verbose logs.
                - 0: Diffusers/Transformers warning logging on local main process only
                - 1: Diffusers/Transformers info logging on local main process only
                - 2: Diffusers/Transformers debug logging on local main process only
                - 3: Diffusers/Transformers debug logging on all processes
    """

    seed: Optional[int] = None
    tracker_name: str = "finetrainers-inference"
    output_dir: str = None
    logging_dir: Optional[str] = "logs"
    init_timeout: int = 300  # 5 minutes
    nccl_timeout: int = 600  # 10 minutes
    report_to: str = "wandb"
    verbose: int = 1

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--tracker_name", type=str, default="finetrainers")
        parser.add_argument("--output_dir", type=str, default="finetrainers-inference")
        parser.add_argument("--logging_dir", type=str, default="logs")
        parser.add_argument("--init_timeout", type=int, default=300)
        parser.add_argument("--nccl_timeout", type=int, default=600)
        parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
        parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3])

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.seed = argparse_args.seed
        mapped_args.tracker_name = argparse_args.tracker_name
        mapped_args.output_dir = argparse_args.output_dir
        mapped_args.logging_dir = argparse_args.logging_dir
        mapped_args.init_timeout = argparse_args.init_timeout
        mapped_args.nccl_timeout = argparse_args.nccl_timeout
        mapped_args.report_to = argparse_args.report_to
        mapped_args.verbose = argparse_args.verbose

    def validate_args(self, args: "BaseArgs"):
        pass


class BaseArgs:
    """The arguments for the finetrainers inference script."""

    parallel_args = ParallelArgs()
    model_args = ModelArgs()
    inference_args = InferenceArgs()
    attention_provider_args = AttentionProviderArgs()
    torch_config_args = TorchConfigArgs()
    miscellaneous_args = MiscellaneousArgs()

    _registered_config_mixins: List[ArgsConfigMixin] = []
    _arg_group_map: Dict[str, ArgsConfigMixin] = {}

    def __init__(self):
        self._arg_group_map: Dict[str, ArgsConfigMixin] = {
            "parallel_args": self.parallel_args,
            "model_args": self.model_args,
            "inference_args": self.inference_args,
            "attention_provider_args": self.attention_provider_args,
            "torch_config_args": self.torch_config_args,
            "miscellaneous_args": self.miscellaneous_args,
        }

        for arg_config_mixin in self._arg_group_map.values():
            self.register_args(arg_config_mixin)

    def to_dict(self) -> Dict[str, Any]:
        arguments_to_dict = {}
        for config_mixin in self._registered_config_mixins:
            arguments_to_dict[config_mixin.__class__.__name__] = config_mixin.to_dict()

        return arguments_to_dict

    def register_args(self, config: ArgsConfigMixin) -> None:
        if not hasattr(self, "_extended_add_arguments"):
            self._extended_add_arguments = []
        self._extended_add_arguments.append((config.add_args, config.validate_args, config.map_args))
        self._registered_config_mixins.append(config)

    def parse_args(self):
        parser = argparse.ArgumentParser()

        for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
            add_fn, _, _ = extended_add_arg_fns
            add_fn(parser)

        args, remaining_args = parser.parse_known_args()
        logger.debug(f"Remaining unparsed arguments: {remaining_args}")

        for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
            _, _, map_fn = extended_add_arg_fns
            map_fn(args, self)

        for extended_add_arg_fns in getattr(self, "_extended_add_arguments", []):
            _, validate_fn, _ = extended_add_arg_fns
            validate_fn(self)

        return self

    def __getattribute__(self, name: str):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            for arg_group in self._arg_group_map.values():
                if hasattr(arg_group, name):
                    return getattr(arg_group, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__:
            object.__setattr__(self, name, value)
            return
        for arg_group in self._arg_group_map.values():
            if hasattr(arg_group, name):
                setattr(arg_group, name, value)
                return
        object.__setattr__(self, name, value)


if __name__ == "__main__":
    main()
