import importlib
import json
import os
from typing import Optional

from huggingface_hub import hf_hub_download


def resolve_component_cls(
    pretrained_model_name_or_path: str,
    component_name: str,
    filename: str = "model_index.json",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.exists(str(pretrained_model_name_or_path)) and os.path.isdir(pretrained_model_name_or_path):
        index_path = os.path.join(pretrained_model_name_or_path, filename)
    else:
        index_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=filename, revision=revision, cache_dir=cache_dir
        )

    with open(index_path, "r") as f:
        model_index_dict = json.load(f)

    if component_name not in model_index_dict:
        raise ValueError(f"No {component_name} found in the model index dict.")

    cls_config = model_index_dict[component_name]
    library = importlib.import_module(cls_config[0])
    return getattr(library, cls_config[1])
