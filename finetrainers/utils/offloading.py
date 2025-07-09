from typing import Dict, List, Optional, Union

import torch


# Import diffusers hooks at module level for testing purposes
try:
    from diffusers.hooks import apply_group_offloading
    from diffusers.hooks.group_offloading import _is_group_offload_enabled

    _DIFFUSERS_AVAILABLE = True
except ImportError:
    apply_group_offloading = None
    _is_group_offload_enabled = None
    _DIFFUSERS_AVAILABLE = False


def enable_group_offload_on_components(
    components: Dict[str, torch.nn.Module],
    device: Union[torch.device, str],
    offload_type: str = "block_level",
    num_blocks_per_group: Optional[int] = 1,
    use_stream: bool = False,
    record_stream: bool = False,
    low_cpu_mem_usage: bool = False,
    non_blocking: bool = False,
    offload_to_disk_path: Optional[str] = None,
    excluded_components: List[str] = ["vae", "vqvae"],
    required_import_error_message: str = "Group offloading requires diffusers>=0.33.0",
) -> None:
    """
    Enable group offloading on model components.

    Args:
        components (Dict[str, torch.nn.Module]):
            Dictionary of model components to apply group offloading to.
        device (Union[torch.device, str]):
            The device to which the group of modules are onloaded.
        offload_type (str, defaults to "block_level"):
            The type of offloading to be applied. Can be one of "block_level" or "leaf_level".
        num_blocks_per_group (int, optional, defaults to 1):
            The number of blocks per group when using offload_type="block_level".
        use_stream (bool, defaults to False):
            If True, offloading and onloading is done asynchronously using a CUDA stream.
        record_stream (bool, defaults to False):
            When enabled with `use_stream`, it marks the tensor as having been used by this stream.
        low_cpu_mem_usage (bool, defaults to False):
            If True, CPU memory usage is minimized by pinning tensors on-the-fly instead of pre-pinning them.
        non_blocking (bool, defaults to False):
            If True, offloading and onloading is done with non-blocking data transfer.
        offload_to_disk_path (str, optional, defaults to None):
            The path to the directory where parameters will be offloaded to disk.
        excluded_components (List[str], defaults to ["vae", "vqvae"]):
            List of component names to exclude from group offloading.
        required_import_error_message (str, defaults to "Group offloading requires diffusers>=0.33.0"):
            Error message to display when required imports are not available.
    """
    if not _DIFFUSERS_AVAILABLE:
        raise ImportError(required_import_error_message)

    onload_device = torch.device(device)
    offload_device = torch.device("cpu")

    for name, component in components.items():
        if name in excluded_components:
            # Skip excluded components
            component.to(onload_device)
            continue

        if not isinstance(component, torch.nn.Module):
            continue

        # Skip components that already have group offloading enabled
        if _is_group_offload_enabled(component):
            continue

        # Apply group offloading based on whether the component has the ModelMixin interface
        if hasattr(component, "enable_group_offload"):
            # For diffusers ModelMixin implementations
            kwargs = {
                "onload_device": onload_device,
                "offload_device": offload_device,
                "offload_type": offload_type,
                "use_stream": use_stream,
                "record_stream": record_stream,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "non_blocking": non_blocking,
                "offload_to_disk_path": offload_to_disk_path,
            }
            if offload_type == "block_level" and num_blocks_per_group is not None:
                kwargs["num_blocks_per_group"] = num_blocks_per_group

            component.enable_group_offload(**kwargs)
        else:
            # For other torch.nn.Module implementations
            kwargs = {
                "module": component,
                "onload_device": onload_device,
                "offload_device": offload_device,
                "offload_type": offload_type,
                "use_stream": use_stream,
                "record_stream": record_stream,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "non_blocking": non_blocking,
                "offload_to_disk_path": offload_to_disk_path,
            }
            if offload_type == "block_level" and num_blocks_per_group is not None:
                kwargs["num_blocks_per_group"] = num_blocks_per_group

            apply_group_offloading(**kwargs)
