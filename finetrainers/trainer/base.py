import contextlib
import functools
from typing import Callable, List, Tuple

import torch
from diffusers.hooks import HookRegistry, ModelHook

from finetrainers.args import BaseArgsType
from finetrainers.logging import get_logger
from finetrainers.models.attention_dispatch import AttentionProvider, _AttentionProviderRegistry


logger = get_logger()

_LATEST_ACTIVE_MODULE_HOOK = "latest_active_module_hook"


class Trainer:
    def __init__(self, args: BaseArgsType):
        self.args = args

        self._module_name_providers_training = _parse_attention_providers(args.attn_provider_training)
        self._module_name_providers_inference = _parse_attention_providers(args.attn_provider_inference)

    @contextlib.contextmanager
    def attention_provider_ctx(self, training: bool = True):
        name_providers_active = (
            self._module_name_providers_training if training else self._module_name_providers_inference
        )
        name_providers_dict = dict(name_providers_active)
        default_provider = _AttentionProviderRegistry._active_provider

        all_registered_module_names = [
            attr for attr in dir(self) if isinstance(getattr(self, attr, None), torch.nn.Module)
        ]
        for module_name in all_registered_module_names:
            if module_name in name_providers_dict:
                continue
            name_providers_dict[module_name] = default_provider

        module_providers_dict = {}
        for module_name, provider in name_providers_dict.items():
            module = getattr(self, module_name, None)
            if module is not None:
                module_providers_dict[module] = (module_name, provider)

        # We don't want to immediately unset the attention provider to default after forward because if the
        # model is being trained, the backward pass must be invoked with the same attention provider
        # So, we lazily switch attention providers only when the forward pass of a new module is called
        def callback(m: torch.nn.Module):
            module_name, provider = module_providers_dict[m]
            _AttentionProviderRegistry._active_provider = provider

        # HACK: for VAE
        if "vae" in name_providers_dict:
            _apply_forward_hooks_hack(self.vae, name_providers_dict["vae"])

        for module in module_providers_dict.keys():
            registry = HookRegistry.check_if_exists_or_initialize(module)
            hook = LatestActiveModuleHook(callback)
            registry.register_hook(hook, _LATEST_ACTIVE_MODULE_HOOK)

        yield

        _AttentionProviderRegistry._active_provider = default_provider
        for module in module_providers_dict.keys():
            registry: HookRegistry = module._diffusers_hook
            registry.remove_hook(_LATEST_ACTIVE_MODULE_HOOK)


class LatestActiveModuleHook(ModelHook):
    def __init__(self, callback: Callable[[torch.nn.Module], None] = None):
        super().__init__()
        self.callback = callback

    def pre_forward(self, module, *args, **kwargs):
        self.callback(module)
        return args, kwargs


def _parse_attention_providers(attn_providers: List[str] = None) -> List[Tuple[str, AttentionProvider]]:
    parsed_providers = []
    if attn_providers:
        for provider_str in attn_providers:
            parts = provider_str.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid attention provider format: '{provider_str}'. Expected 'module_name:provider_name'."
                )
            parts[1] = AttentionProvider(parts[1])
            parsed_providers.append(tuple(parts))
    return parsed_providers


def _apply_forward_hooks_hack(module: torch.nn.Module, provider: AttentionProvider):
    if hasattr(module, "_finetrainers_wrapped_methods"):
        return

    def create_wrapper(old_method):
        @functools.wraps(old_method)
        def wrapper(*args, **kwargs):
            old_provider = _AttentionProviderRegistry._active_provider
            _AttentionProviderRegistry._active_provider = provider
            output = old_method(*args, **kwargs)
            _AttentionProviderRegistry._active_provider = old_provider
            return output

        return wrapper

    methods = ["encode", "decode", "_encode", "_decode", "tiled_encode", "tiled_decode"]
    finetrainers_wrapped_methods = []
    for method_name in methods:
        if not hasattr(module, method_name):
            continue
        method = getattr(module, method_name)
        wrapper = create_wrapper(method)
        setattr(module, method_name, wrapper)
        finetrainers_wrapped_methods.append(method_name)
    module._finetrainers_wrapped_methods = finetrainers_wrapped_methods
