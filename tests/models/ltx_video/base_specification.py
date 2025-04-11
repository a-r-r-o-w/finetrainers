import torch
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXVideoTransformer3DModel
from transformers import AutoTokenizer, T5EncoderModel

from finetrainers.models.ltx_video import LTXVideoModelSpecification


class DummyLTXVideoModelSpecification(LTXVideoModelSpecification):
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
        vae = AutoencoderKLLTXVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 8, 8, 8),
            layers_per_block=(1, 1, 1, 1, 1),
            decoder_layers_per_block=(1, 1, 1, 1, 1),
            spatio_temporal_scaling=(True, True, False, False),
            decoder_spatio_temporal_scaling=(True, True, False, False),
            decoder_inject_noise=(False, False, False, False, False),
            upsample_residual=(False, False, False, False),
            upsample_factor=(1, 1, 1, 1),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        vae.to(self.vae_dtype)
        self.vae_config = vae.config
        return {"vae": vae}

    def load_diffusion_models(self):
        torch.manual_seed(0)
        transformer = LTXVideoTransformer3DModel(
            in_channels=8,
            out_channels=8,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=4,
            attention_head_dim=8,
            cross_attention_dim=32,
            num_layers=1,
            caption_channels=32,
        )
        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        transformer.to(self.transformer_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler()
        return {"transformer": transformer, "scheduler": scheduler}
