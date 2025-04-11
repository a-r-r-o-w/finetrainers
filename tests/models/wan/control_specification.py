import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
from transformers import AutoTokenizer, T5EncoderModel

from finetrainers.models.utils import _expand_conv3d_with_zeroed_weights
from finetrainers.models.wan import WanControlModelSpecification


class DummyWanControlModelSpecification(WanControlModelSpecification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # This needs to be updated for the test to work correctly.
        # TODO(aryan): it will not be needed if we hosted the dummy model so that the correct config could be loaded
        # with ModelSpecification::_load_configs
        self.transformer_config.in_channels = 16

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

    def load_diffusion_models(self, new_in_features: int):
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
        ).to(self.transformer_dtype)

        transformer.patch_embedding = _expand_conv3d_with_zeroed_weights(
            transformer.patch_embedding, new_in_channels=new_in_features
        )
        transformer.register_to_config(in_channels=new_in_features)

        # TODO(aryan): Upload dummy checkpoints to the Hub so that we don't have to do this.
        # Doing so overrides things like _keep_in_fp32_modules
        transformer.to(self.transformer_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler()
        return {"transformer": transformer, "scheduler": scheduler}
