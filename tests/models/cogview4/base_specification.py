import pathlib
import sys

import torch
from diffusers import AutoencoderKL, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, GlmModel


project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from finetrainers.models.cogview4 import CogView4ModelSpecification  # noqa


class DummyCogView4ModelSpecification(CogView4ModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_condition_models(self):
        text_encoder = GlmModel.from_pretrained(
            "hf-internal-testing/tiny-random-cogview4", subfolder="text_encoder", torch_dtype=self.text_encoder_dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-cogview4", subfolder="tokenizer", trust_remote_code=True
        )
        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    def load_latent_models(self):
        torch.manual_seed(0)
        vae = AutoencoderKL.from_pretrained(
            "hf-internal-testing/tiny-random-cogview4", subfolder="vae", torch_dtype=self.vae_dtype
        )
        self.vae_config = vae.config
        return {"vae": vae}

    def load_diffusion_models(self):
        torch.manual_seed(0)
        transformer = CogView4Transformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-random-cogview4", subfolder="transformer", torch_dtype=self.transformer_dtype
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        return {"transformer": transformer, "scheduler": scheduler}
