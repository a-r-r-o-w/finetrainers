from diffusers import (
    CogVideoXTransformer3DModel,
    CogView4Transformer2DModel,
    FluxTransformer2DModel,
    WanTransformer3DModel,
)

from finetrainers._metadata import CPInput, CPOutput, ParamId, TransformerMetadata, TransformerRegistry
from finetrainers.logging import get_logger


logger = get_logger()


def register_transformer_metadata():
    # CogVideoX
    TransformerRegistry.register(
        model_class=CogVideoXTransformer3DModel,
        metadata=TransformerMetadata(
            cp_plan={
                "": {
                    ParamId("image_rotary_emb", 5): [CPInput(0, 2), CPInput(0, 2)],
                },
                "transformer_blocks.0": {
                    ParamId("hidden_states", 0): CPInput(1, 3),
                    ParamId("encoder_hidden_states", 1): CPInput(1, 3),
                },
                "proj_out": [CPOutput(1, 3)],
            }
        ),
    )

    # CogView4
    TransformerRegistry.register(
        model_class=CogView4Transformer2DModel,
        metadata=TransformerMetadata(
            cp_plan={
                "patch_embed": {
                    ParamId(index=0): CPInput(1, 3, split_output=True),
                    ParamId(index=1): CPInput(1, 3, split_output=True),
                },
                "rope": {
                    ParamId(index=0): CPInput(0, 2, split_output=True),
                    ParamId(index=1): CPInput(0, 2, split_output=True),
                },
                "proj_out": [CPOutput(1, 3)],
            }
        ),
    )

    # Flux
    TransformerRegistry.register(
        model_class=FluxTransformer2DModel,
        metadata=TransformerMetadata(
            cp_plan={
                "": {
                    ParamId("hidden_states", 0): CPInput(1, 3),
                    ParamId("encoder_hidden_states", 1): CPInput(1, 3),
                    ParamId("img_ids", 4): CPInput(0, 2),
                    ParamId("txt_ids", 5): CPInput(0, 2),
                },
                "proj_out": [CPOutput(1, 3)],
            }
        ),
    )

    # Wan2.1
    TransformerRegistry.register(
        model_class=WanTransformer3DModel,
        metadata=TransformerMetadata(
            cp_plan={
                "rope": {
                    ParamId(index=0): CPInput(2, 4, split_output=True),
                },
                "blocks.*": {
                    ParamId("encoder_hidden_states", 1): CPInput(1, 3),
                },
                "blocks.0": {
                    ParamId("hidden_states", 0): CPInput(1, 3),
                },
                "proj_out": [CPOutput(1, 3)],
            }
        ),
    )

    logger.debug("Metadata for transformer registered")
