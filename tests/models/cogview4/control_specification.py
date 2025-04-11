import torch
from diffusers import AutoencoderKL, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, GlmConfig, GlmModel

from finetrainers.models.cogview4 import CogView4ControlModelSpecification
from finetrainers.models.utils import _expand_linear_with_zeroed_weights


class DummyCogView4ControlModelSpecification(CogView4ControlModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # This needs to be updated for the test to work correctly.
        # TODO(aryan): it will not be needed if we hosted the dummy model so that the correct config could be loaded
        # with ModelSpecification::_load_configs
        self.transformer_config.in_channels = 4

    def load_condition_models(self):
        text_encoder_config = GlmConfig(
            hidden_size=32, intermediate_size=8, num_hidden_layers=2, num_attention_heads=4, head_dim=8
        )
        text_encoder = GlmModel(text_encoder_config).to(self.text_encoder_dtype)
        # TODO(aryan): try to not rely on trust_remote_code by creating dummy tokenizer
        tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    def load_latent_models(self):
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        ).to(self.vae_dtype)
        return {"vae": vae}

    def load_diffusion_models(self, new_in_features: int):
        torch.manual_seed(0)
        transformer = CogView4Transformer2DModel(
            patch_size=2,
            in_channels=4,
            num_layers=2,
            attention_head_dim=4,
            num_attention_heads=4,
            out_channels=4,
            text_embed_dim=32,
            time_embed_dim=8,
            condition_dim=4,
        ).to(self.transformer_dtype)
        actual_new_in_features = new_in_features * transformer.config.patch_size**2
        transformer.patch_embed.proj = _expand_linear_with_zeroed_weights(
            transformer.patch_embed.proj, new_in_features=actual_new_in_features
        )
        transformer.register_to_config(in_channels=new_in_features)

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}
