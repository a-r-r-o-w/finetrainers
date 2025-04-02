from typing import Dict, Optional, Any
import torch
from torch.nn import Module
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, LlamaModel
from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,)

from ...utils import _enable_vae_memory_optimizations, get_non_null_items

def load_condition_models(self) -> Dict[str, torch.nn.Module]:
    common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

    if self.tokenizer_id is not None:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
        )

    if self.tokenizer_2_id is not None:
        tokenizer_2 = AutoTokenizer.from_pretrained(self.tokenizer_2_id, **common_kwargs)
    else:
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer_2", **common_kwargs
        )

    if self.text_encoder_id is not None:
        text_encoder = LlamaModel.from_pretrained(
            self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
        )
    else:
        text_encoder = LlamaModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=self.text_encoder_dtype,
            **common_kwargs,
        )

    if self.text_encoder_2_id is not None:
        text_encoder_2 = CLIPTextModel.from_pretrained(
            self.text_encoder_2_id, torch_dtype=self.text_encoder_2_dtype, **common_kwargs
        )
    else:
        text_encoder_2 = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=self.text_encoder_2_dtype,
            **common_kwargs,
        )

    return {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
    }

def load_latent_models(self) -> Dict[str, torch.nn.Module]:
    common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

    if self.vae_id is not None:
        vae = AutoencoderKLHunyuanVideo.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
    else:
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
        )

    return {"vae": vae}

def load_pipeline(
    self,
    tokenizer: Optional[AutoTokenizer] = None,
    tokenizer_2: Optional[CLIPTokenizer] = None,
    text_encoder: Optional[LlamaModel] = None,
    text_encoder_2: Optional[CLIPTextModel] = None,
    transformer: Optional[Module] = None,
    vae: Optional[AutoencoderKLHunyuanVideo] = None,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    training: bool = False,
    **kwargs,
) -> HunyuanVideoPipeline:
    components = {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "transformer": transformer,
        "vae": vae,
        "scheduler": scheduler,
    }
    components = get_non_null_items(components)

    pipe = HunyuanVideoPipeline.from_pretrained(
        self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
    )
    pipe.text_encoder.to(self.text_encoder_dtype)
    pipe.text_encoder_2.to(self.text_encoder_2_dtype)
    pipe.vae.to(self.vae_dtype)

    _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
    if not training:
        pipe.transformer.to(self.transformer_dtype)
