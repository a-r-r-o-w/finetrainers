from huggingface_hub import hf_hub_download 
import os 
import importlib
import json 

def _resolve_vae_cls_from_ckpt_path(ckpt_path):
    # TODO: consider accepting revision.
    ckpt_path = str(ckpt_path)
    if os.path.exists(str(ckpt_path)) and os.path.isdir(ckpt_path):
        index_path = os.path.join(ckpt_path, "model_index.json")
    else:
        index_path = hf_hub_download(repo_id=ckpt_path, filename="model_index.json")
    
    with open(index_path, "r") as f:
        model_index_dict = json.load(f)
    assert "vae" in model_index_dict, "No VAE found in the modelx index dict."
    
    vae_cls_config = model_index_dict["vae"]
    library =  importlib.import_module(vae_cls_config[0])
    return getattr(library, vae_cls_config[1])
