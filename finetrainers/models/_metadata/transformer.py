from diffusers import WanTransformer3DModel

from finetrainers._metadata import (
    ContextParallelInputMetadata,
    ContextParallelOutputMetadata,
    ParamIdentifier,
    TransformerMetadata,
    TransformerRegistry,
)
from finetrainers.logging import get_logger


logger = get_logger()


# ===== Transformer Metadata Registrations =====


def register_transformer_metadata():
    # Wan2.1
    TransformerRegistry.register(
        model_class=WanTransformer3DModel,
        metadata=TransformerMetadata(
            cp_plan={
                # NOTE: this is probably suboptimal since we shard at every layer. The overhead should be minimal
                # but might be slower and worth investigating.
                "blocks.*": {
                    ParamIdentifier("encoder_hidden_states", 1): ContextParallelInputMetadata(1, 3),
                    ParamIdentifier("rotary_emb", 3): ContextParallelInputMetadata(2, 4),
                },
                "blocks.0": {
                    ParamIdentifier("hidden_states", 0): ContextParallelInputMetadata(1, 3),
                },
                "proj_out": [ContextParallelOutputMetadata(1, 3)],
            }
        ),
    )

    logger.debug("Metadata for transformer registered")
