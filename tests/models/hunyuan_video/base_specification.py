import pathlib
import sys

import torch
from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler, HunyuanVideoTransformer3DModel
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
)


project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from finetrainers.models.hunyuan_video import HunyuanVideoModelSpecification  # noqa


class DummyHunyuanVideoModelSpecification(HunyuanVideoModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_condition_models(self):
        llama_text_encoder_config = LlamaConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=8,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = LlamaModel(llama_text_encoder_config)
        tokenizer = LlamaTokenizer.from_pretrained("finetrainers/dummy-hunyaunvideo", subfolder="tokenizer")

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModel(clip_text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder.to(self.text_encoder_dtype)
        text_encoder_2.to(self.text_encoder_2_dtype)

        return {
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
        }

    def load_latent_models(self):
        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=(
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            up_block_types=(
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=4,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
            mid_block_add_attention=True,
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        vae.to(self.vae_dtype)
        self.vae_config = vae.config
        return {"vae": vae}

    def load_diffusion_models(self):
        torch.manual_seed(0)
        transformer = HunyuanVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=10,
            num_layers=2,
            num_single_layers=2,
            num_refiner_layers=1,
            patch_size=1,
            patch_size_t=1,
            guidance_embeds=True,
            text_embed_dim=16,
            pooled_projection_dim=8,
            rope_axes_dim=(2, 4, 4),
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        transformer.to(self.transformer_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler()
        return {"transformer": transformer, "scheduler": scheduler}
