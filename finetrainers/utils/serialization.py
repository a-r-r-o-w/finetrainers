from typing import Any, Dict, Optional

import safetensors.torch


def safetensors_torch_save_function(weights: Dict[str, Any], filename: str, metadata: Optional[Dict[str, str]] = None):
    if metadata is None:
        metadata = {}
    metadata["format"] = "pt"
    safetensors.torch.save_file(weights, filename, metadata)
