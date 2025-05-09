from typing import TYPE_CHECKING

import torch

from .dependencies.diffusers.peft import load_lora_weights


if TYPE_CHECKING:
    from finetrainers.args import BaseArgsType
    from finetrainers.parallel import ParallelBackendType


def perform_patches_for_training(args: "BaseArgsType", parallel_backend: "ParallelBackendType") -> None:
    # To avoid circular imports
    from finetrainers.config import ModelType, TrainingType

    from .dependencies.diffusers import patch

    # Modeling patches
    patch_scaled_dot_product_attention()

    patch.patch_diffusers_rms_norm_forward()

    # LTX Video patches
    if args.model_name == ModelType.LTX_VIDEO:
        from .models.ltx_video import patch

        patch.patch_transformer_forward()
        if parallel_backend.tensor_parallel_enabled:
            patch.patch_apply_rotary_emb_for_tp_compatibility()

    # Wan patches
    if args.model_name == ModelType.WAN and "transformer" in args.layerwise_upcasting_modules:
        from .models.wan import patch

        patch.patch_time_text_image_embedding_forward()

    # LoRA patches
    if args.training_type == TrainingType.LORA and len(args.layerwise_upcasting_modules) > 0:
        from .dependencies.peft import patch

        patch.patch_peft_move_adapter_to_device_of_base_layer()


def perform_patches_for_inference(args: "BaseArgsType", parallel_backend: "ParallelBackendType") -> None:
    # To avoid circular imports
    from .dependencies.diffusers import patch

    # Modeling patches
    patch_scaled_dot_product_attention()

    patch.patch_diffusers_rms_norm_forward()


def patch_scaled_dot_product_attention():
    from finetrainers.models.attention_dispatch import attention_dispatch

    torch.nn.functional.scaled_dot_product_attention = attention_dispatch
