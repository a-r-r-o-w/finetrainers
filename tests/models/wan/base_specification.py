import pathlib
import sys

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
from transformers import AutoTokenizer, T5EncoderModel


project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from finetrainers.models.wan import WanModelSpecification  # noqa


class DummyWanModelSpecification(WanModelSpecification):
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
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        vae.to(self.vae_dtype)
        self.vae_config = vae.config
        return {"vae": vae}

    def load_diffusion_models(self):
        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        transformer.to(self.transformer_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler()
        return {"transformer": transformer, "scheduler": scheduler}
