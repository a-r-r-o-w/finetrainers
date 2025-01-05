import importlib
import json
import os

from huggingface_hub import hf_hub_download


def resolve_vae_cls_from_ckpt_path(ckpt_path, **kwargs):
    ckpt_path = str(ckpt_path)
    if os.path.exists(str(ckpt_path)) and os.path.isdir(ckpt_path):
        index_path = os.path.join(ckpt_path, "model_index.json")
    else:
        revision = kwargs.get("revision", None)
        cache_dir = kwargs.get("cache_dir", None)
        index_path = hf_hub_download(
            repo_id=ckpt_path, filename="model_index.json", revision=revision, cache_dir=cache_dir
        )

    with open(index_path, "r") as f:
        model_index_dict = json.load(f)
    assert "vae" in model_index_dict, "No VAE found in the modelx index dict."

    vae_cls_config = model_index_dict["vae"]
    library = importlib.import_module(vae_cls_config[0])
    return getattr(library, vae_cls_config[1])
