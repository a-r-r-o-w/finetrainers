import contextlib
import functools
import os
from typing import Callable, List, Tuple

import torch
import torch.backends
from diffusers.hooks import HookRegistry, ModelHook

from finetrainers import logging, parallel, patches
from finetrainers.args import BaseArgsType
from finetrainers.logging import get_logger
from finetrainers.models.attention_dispatch import AttentionProvider, _AttentionProviderRegistry
from finetrainers.state import State


logger = get_logger()

_LATEST_ACTIVE_MODULE_HOOK = "latest_active_module_hook"


class Trainer:
    def __init__(self, args: BaseArgsType):
        self.args = args

        self.state = State()

        self._module_name_providers_training = _parse_attention_providers(args.attn_provider_training)
        self._module_name_providers_inference = _parse_attention_providers(args.attn_provider_inference)

        self._init_distributed()
        self._init_config_options()

        # Perform any patches that might be necessary for training to work as expected
        patches.perform_patches_for_training(self.args, self.state.parallel_backend)

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
            # HACK: for CP on transformer. Need to support other modules too and improve overall experience for external usage
            if module_name in ["transformer"] and self.state.parallel_backend.context_parallel_enabled:
                if not _AttentionProviderRegistry.supports_context_parallel(provider):
                    raise ValueError(
                        f"Attention provider {provider} does not support context parallel. Please use a different provider."
                    )
                _AttentionProviderRegistry._set_context_parallel(
                    mesh=self.state.parallel_backend.get_mesh()["cp"], convert_to_fp32=True, rotate_method="allgather"
                )
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
        _AttentionProviderRegistry._set_context_parallel(reset=True)
        for module in module_providers_dict.keys():
            registry: HookRegistry = module._diffusers_hook
            registry.remove_hook(_LATEST_ACTIVE_MODULE_HOOK)

    def _init_distributed(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        # TODO(aryan): handle other backends
        backend_cls: parallel.ParallelBackendType = parallel.get_parallel_backend_cls(self.args.parallel_backend)
        self.state.parallel_backend = backend_cls(
            world_size=world_size,
            pp_degree=self.args.pp_degree,
            dp_degree=self.args.dp_degree,
            dp_shards=self.args.dp_shards,
            cp_degree=self.args.cp_degree,
            tp_degree=self.args.tp_degree,
            backend="nccl",
            timeout=self.args.init_timeout,
            logging_dir=self.args.logging_dir,
            output_dir=self.args.output_dir,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        if self.args.seed is not None:
            self.state.parallel_backend.enable_determinism(self.args.seed)

    def _init_logging(self) -> None:
        logging._set_parallel_backend(self.state.parallel_backend)
        logging.set_dependency_log_level(self.args.verbose, self.state.parallel_backend.is_local_main_process)
        logger.info("Initialized FineTrainers")

    def _init_trackers(self) -> None:
        # TODO(aryan): handle multiple trackers
        trackers = [self.args.report_to]
        experiment_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.parallel_backend.initialize_trackers(
            trackers, experiment_name=experiment_name, config=self._get_training_info(), log_dir=self.args.logging_dir
        )

    def _init_config_options(self) -> None:
        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision(self.args.float32_matmul_precision)

    @property
    def tracker(self):
        return self.state.parallel_backend.tracker


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


# TODO(aryan): instead of this, we could probably just apply the hook to vae.children() as we know their forward methods will be invoked
def _apply_forward_hooks_hack(module: torch.nn.Module, provider: AttentionProvider):
    if hasattr(module, "_finetrainers_wrapped_methods"):
        return

    def create_wrapper(old_method):
        @functools.wraps(old_method)
        def wrapper(*args, **kwargs):
            _AttentionProviderRegistry._set_context_parallel(reset=True)  # HACK: needs improvement
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
