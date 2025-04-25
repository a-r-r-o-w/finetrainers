from diffusers import FluxTransformer2DModel, WanTransformer3DModel

from finetrainers._metadata import (
    ContextParallelInputMetadata,
    ContextParallelOutputMetadata,
    ParamIdentifier,
    TransformerMetadata,
    TransformerRegistry,
)
from finetrainers.logging import get_logger


logger = get_logger()


def register_transformer_metadata():
    # Flux
    TransformerRegistry.register(
        model_class=FluxTransformer2DModel,
        metadata=TransformerMetadata(
            cp_plan={
                "": {
                    ParamIdentifier("hidden_states", 0): ContextParallelInputMetadata(1, 3),
                    ParamIdentifier("encoder_hidden_states", 1): ContextParallelInputMetadata(1, 3),
                    ParamIdentifier("img_ids", 4): ContextParallelInputMetadata(0, 2),
                    ParamIdentifier("txt_ids", 5): ContextParallelInputMetadata(0, 2),
                },
                "proj_out": [ContextParallelOutputMetadata(1, 3)],
            }
        ),
    )

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
