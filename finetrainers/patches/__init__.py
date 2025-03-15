from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..args import BaseArgs
    from ..parallel import ParallelBackendType


def perform_patches_for_training(args: "BaseArgs", parallel_backend: "ParallelBackendType") -> None:
    # To avoid circular imports
    from ..config import ModelType, TrainingType

    if args.model_name == ModelType.LTX_VIDEO:
        from .models.ltx_video import patch

        patch.patch_transformer_forward()
        if parallel_backend.tensor_parallel_enabled:
            patch.patch_apply_rotary_emb_for_tp_compatibility()

    if args.model_name == ModelType.WAN and "transformer" in args.layerwise_upcasting_modules:
        from .models.wan import patch

        patch.patch_time_text_image_embedding_forward()

    if args.training_type == TrainingType.LORA and len(args.layerwise_upcasting_modules) > 0:
        from .dependencies.peft import patch

        patch.patch_peft_move_adapter_to_device_of_base_layer()
