import pathlib
import sys

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXTransformer3DModel
from transformers import AutoTokenizer, T5EncoderModel


project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from finetrainers.models.cogvideox import CogVideoXModelSpecification  # noqa


class DummyCogVideoXModelSpecification(CogVideoXModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_condition_models(self):
        text_encoder = T5EncoderModel.from_pretrained(
            "hf-internal-testing/tiny-random-t5", torch_dtype=self.text_encoder_dtype
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    def load_latent_models(self):
        torch.manual_seed(0)
        vae = AutoencoderKLCogVideoX(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ),
            up_block_types=(
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=2,
            temporal_compression_ratio=4,
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        vae.to(self.vae_dtype)
        self.vae_config = vae.config
        return {"vae": vae}

    def load_diffusion_models(self):
        torch.manual_seed(0)
        transformer = CogVideoXTransformer3DModel(
            num_attention_heads=4,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            time_embed_dim=2,
            text_embed_dim=32,
            num_layers=2,
            sample_width=24,
            sample_height=24,
            sample_frames=9,
            patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=16,
            use_rotary_positional_embeddings=True,
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        transformer.to(self.transformer_dtype)
        self.transformer_config = transformer.config
        scheduler = CogVideoXDDIMScheduler()
        return {"transformer": transformer, "scheduler": scheduler}
