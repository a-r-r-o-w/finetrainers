from diffusers import WanTransformer3DModel

from finetrainers._metadata import (
    ContextParallelInputMetadata,
    ContextParallelOutputMetadata,
    ParamIdentifier,
    TransformerMetadata,
    TransformerRegistry,
)


# ===== Metadata Registrations =====

# Wan2.1
TransformerRegistry.register(
    model_class=WanTransformer3DModel,
    metadata=TransformerMetadata(
        cp_plan={
            "blocks.0": {
                ParamIdentifier("hidden_states", 0): ContextParallelInputMetadata(1, 3),
                ParamIdentifier("encoder_hidden_states", 1): ContextParallelInputMetadata(1, 3),
                ParamIdentifier("rotary_emb", 3): ContextParallelInputMetadata(2, 4),
            },
            "proj_out": [ContextParallelOutputMetadata(1, 3)],
        }
    ),
)
