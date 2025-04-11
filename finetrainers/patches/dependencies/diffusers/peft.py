import json
from pathlib import Path
from typing import Optional

import safetensors.torch
from diffusers import DiffusionPipeline
from diffusers.loaders.lora_pipeline import _LOW_CPU_MEM_USAGE_DEFAULT_LORA
from huggingface_hub import repo_exists, snapshot_download
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

from finetrainers.logging import get_logger
from finetrainers.utils import find_files


logger = get_logger()


def load_lora_weights(
    pipeline: DiffusionPipeline, pretrained_model_name_or_path: str, adapter_name: Optional[str] = None, **kwargs
) -> None:
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)

    is_local_file_path = Path(pretrained_model_name_or_path).is_dir()
    if not is_local_file_path:
        does_repo_exist = repo_exists(pretrained_model_name_or_path, repo_type="model")
        if not does_repo_exist:
            raise ValueError(f"Model repo {pretrained_model_name_or_path} does not exist on the Hub or locally.")
        else:
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")

    prefix = "transformer"
    state_dict = pipeline.lora_state_dict(pretrained_model_name_or_path)
    state_dict = {k[len(f"{prefix}.") :]: v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}

    file_list = find_files(pretrained_model_name_or_path, "*.safetensors", depth=1)
    if len(file_list) == 0:
        raise ValueError(f"No .safetensors files found in {pretrained_model_name_or_path}.")
    if len(file_list) > 1:
        logger.warning(
            f"Multiple .safetensors files found in {pretrained_model_name_or_path}. Using the first one: {file_list[0]}."
        )
    with safetensors.torch.safe_open(file_list[0], framework="pt") as f:
        metadata = f.metadata()
        metadata = json.loads(metadata["lora_config"])

    transformer = pipeline.transformer
    if adapter_name is None:
        adapter_name = "default"

    lora_config = LoraConfig(**metadata)
    inject_adapter_in_model(lora_config, transformer, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
    result = set_peft_model_state_dict(
        transformer,
        state_dict,
        adapter_name=adapter_name,
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    logger.debug(
        f"Loaded LoRA weights from {pretrained_model_name_or_path} into {pipeline.__class__.__name__}. Result: {result}"
    )
