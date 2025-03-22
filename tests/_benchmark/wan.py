import argparse
import copy
import json

import torch
import torch._inductor.config
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from torch._utils import _get_device_module
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    RowwiseParallel,
)

from finetrainers.patches.models.ltx_video import patch
from finetrainers.utils import apply_activation_checkpointing


patch.patch_apply_rotary_emb_for_tp_compatibility()


DEVICE_TYPE = "cuda"
PG_BACKEND = "nccl"
DEVICE_COUNT = _get_device_module(DEVICE_TYPE).device_count()


def benchmark(model, inputs, mode="forward", warmup_steps=2, benchmark_steps=3, precision=3):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warm-up phase to stabilize GPU execution
    for _ in range(warmup_steps):
        if mode == "forward":
            with torch.no_grad():
                _ = model(**inputs)[0]
        else:
            output = model(**inputs)[0]
            loss = output.mean()
            loss.requires_grad = True
            loss.backward()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(benchmark_steps):
        torch.cuda.synchronize()
        start_event.record()

        if mode == "forward":
            with torch.no_grad():
                output = model(**inputs)[0]
        elif mode == "backward":
            output = model(**inputs)[0]
            loss = output.mean()
            loss.requires_grad = True
            loss.backward()
        elif mode == "forward_backward":
            output = model(**inputs)[0]
            loss = output.mean()
            loss.requires_grad = True
            loss.backward()

        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # Time in milliseconds

    avg_time = sum(times) / len(times)
    max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    return round(avg_time, precision), round(max_memory, precision)


def verify_numerical_correctness_and_debug(model, model_tp, inputs, rank, dtype, debug):
    with torch.no_grad():
        output_ref = model(**inputs)[0]
        output_tp = model_tp(**inputs)[0]

    if isinstance(output_tp, DTensor):
        output_tp = output_tp.redistribute(output_tp.device_mesh, [Replicate()]).to_local()

    if dtype == torch.float32:
        is_close = torch.allclose(output_ref, output_tp, atol=1e-4, rtol=1e-4)
    elif dtype == torch.bfloat16:
        is_close = torch.allclose(output_ref, output_tp, atol=3e-2, rtol=3e-2)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    max_diff = (output_ref - output_tp).abs().max().item()

    if rank == 0:
        print(f"Numerical correctness check: {is_close}")
        print(f"Max absolute difference: {max_diff}")

    if debug:
        with torch.no_grad():
            comm_mode = CommDebugMode()
            with comm_mode:
                _ = model_tp(**inputs)[0]

            if rank == 0:
                print()
                print("get_comm_counts:", comm_mode.get_comm_counts())
                # print()
                # print("get_parameter_info:", comm_mode.get_parameter_info())  # Too much noise
                print()
                print("Sharding info:\n" + "".join(f"{k} - {v}\n" for k, v in comm_mode.get_sharding_info().items()))
                print()
                print("get_total_counts:", comm_mode.get_total_counts())
                comm_mode.generate_json_dump("dump_comm_mode_log.json", noise_level=1)
                comm_mode.log_comm_debug_tracing_table_to_file("dump_comm_mode_tracing_table.txt", noise_level=1)


def tp_parallelize(model, device_mesh, tp_plan):
    if tp_plan == "only_ff":
        for block in model.blocks:
            block_tp_plan = {
                "ffn.net.0.proj": ColwiseParallel(),
                "ffn.net.2": RowwiseParallel(),
            }
            parallelize_module(block, device_mesh, block_tp_plan)
        parallelize_module(model, device_mesh, {})

    elif tp_plan == "ff_and_attn":
        for block in model.blocks:
            block_tp_plan = {
                "attn1.to_q": ColwiseParallel(output_layouts=Replicate()),
                "attn1.to_k": ColwiseParallel(output_layouts=Replicate()),
                "attn1.to_v": ColwiseParallel(output_layouts=Replicate()),
                "attn1.to_out.0": RowwiseParallel(input_layouts=Replicate()),
                "attn2.to_q": ColwiseParallel(output_layouts=Replicate()),
                "attn2.to_k": ColwiseParallel(output_layouts=Replicate()),
                "attn2.to_v": ColwiseParallel(output_layouts=Replicate()),
                "attn2.to_out.0": RowwiseParallel(input_layouts=Replicate()),
                "ffn.net.0.proj": ColwiseParallel(),
                "ffn.net.2": RowwiseParallel(),
            }
            parallelize_module(block, device_mesh, block_tp_plan)
        parallelize_module(model, device_mesh, {})


def get_config(model_type):
    if model_type == "Wan-T2V-1.3B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-1.3B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-T2V-14B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-14B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-I2V-14B-480p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-480P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    elif model_type == "Wan-I2V-14B-720p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-720P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
    return config


def run_benchmark(world_size: int, rank: int, args):
    dtype = torch.bfloat16

    if args.async_tp:
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(dist.group.WORLD.group_name)

    config = {
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "cross_attention_dim": 2048,
        "num_layers": 28,
        "activation_fn": "gelu-approximate",
        "qk_norm": "rms_norm_across_heads",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-6,
        "caption_channels": 4096,
        "attention_bias": True,
        "attention_out_bias": True,
    }

    torch.manual_seed(0)
    config = get_config(args.model_type)["diffusers_config"]
    model = WanTransformer3DModel(**config).to(DEVICE_TYPE)
    model.to(dtype)

    model_tp = copy.deepcopy(model)
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))

    tp_parallelize(model_tp, device_mesh, args.tp_plan)

    # Input Tensors
    batch_size = 2
    num_frames, height, width = 49, 512, 832
    spatial_compression_ratio = 8
    temporal_compression_ratio = 4
    latent_num_frames = (num_frames - 1) // temporal_compression_ratio + 1
    latent_height = height // spatial_compression_ratio
    latent_width = width // spatial_compression_ratio
    caption_sequence_length = 256

    requires_grad = args.mode != "forward"
    inputs = {
        "hidden_states": torch.randn(
            batch_size,
            model.config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            device=DEVICE_TYPE,
            dtype=dtype,
            requires_grad=requires_grad,
        ),
        "encoder_hidden_states": torch.randn(
            batch_size, caption_sequence_length, 4096, device=DEVICE_TYPE, dtype=dtype, requires_grad=requires_grad
        ),
        "timestep": torch.randint(
            0, 1000, (batch_size,), device=DEVICE_TYPE, dtype=torch.float32, requires_grad=requires_grad
        ),
        "return_dict": False,
    }

    verify_numerical_correctness_and_debug(model, model_tp, inputs, rank, dtype, args.debug)

    if args.mode != "forward":
        model.train()
        model_tp.train()
        apply_activation_checkpointing(model, "full")
        apply_activation_checkpointing(model_tp, "full")
    else:
        model.eval()
        model_tp.eval()

    model.requires_grad_(requires_grad)
    model_tp.requires_grad_(requires_grad)

    elapsed_single, mem_single = benchmark(
        model, inputs, args.mode, warmup_steps=args.warmup_steps, benchmark_steps=args.benchmark_steps
    )
    elapsed_multi, mem_multi = benchmark(
        model_tp, inputs, args.mode, warmup_steps=args.warmup_steps, benchmark_steps=args.benchmark_steps
    )

    if rank == 0:
        result = {
            "Single GPU": {"Time (ms)": elapsed_single, "Memory (GB)": mem_single},
            "Multi GPU": {"Time (ms)": elapsed_multi, "Memory (GB)": mem_multi},
        }
        print(f"Benchmark ({args.mode}):")
        print(json.dumps(result, indent=4))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="Wan-T2V-1.3B",
        choices=["Wan-T2V-1.3B", "Wan-T2V-14B", "Wan-I2V-14B-480p", "Wan-I2V-14B-720p"],
    )
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward", "forward_backward"])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--benchmark-steps", type=int, default=3)
    parser.add_argument("--tp_plan", type=str, default="only_ff", choices=["only_ff", "ff_and_attn"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--async_tp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dist.init_process_group(PG_BACKEND)
    WORLD_SIZE = dist.get_world_size()
    RANK = dist.get_rank()

    torch.cuda.set_device(RANK)

    if RANK == 0:
        print(f"World size: {WORLD_SIZE}")
        print(f"Device count: {DEVICE_COUNT}")

    try:
        with torch.no_grad():
            run_benchmark(WORLD_SIZE, RANK, args)
    finally:
        dist.destroy_process_group()


# WanTransformer3DModel(
#   (rope): WanRotaryPosEmbed()
#   (patch_embedding): Conv3d(16, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2))
#   (condition_embedder): WanTimeTextImageEmbedding(
#     (timesteps_proj): Timesteps()
#     (time_embedder): TimestepEmbedding(
#       (linear_1): Linear(in_features=256, out_features=1536, bias=True)
#       (act): SiLU()
#       (linear_2): Linear(in_features=1536, out_features=1536, bias=True)
#     )
#     (act_fn): SiLU()
#     (time_proj): Linear(in_features=1536, out_features=9216, bias=True)
#     (text_embedder): PixArtAlphaTextProjection(
#       (linear_1): Linear(in_features=4096, out_features=1536, bias=True)
#       (act_1): GELU(approximate='tanh')
#       (linear_2): Linear(in_features=1536, out_features=1536, bias=True)
#     )
#   )
#   (blocks): ModuleList(
#     (0-29): 30 x WanTransformerBlock(
#       (norm1): FP32LayerNorm((1536,), eps=1e-06, elementwise_affine=False)
#       (attn1): Attention(
#         (norm_q): RMSNorm()
#         (norm_k): RMSNorm()
#         (to_q): Linear(in_features=1536, out_features=1536, bias=True)
#         (to_k): Linear(in_features=1536, out_features=1536, bias=True)
#         (to_v): Linear(in_features=1536, out_features=1536, bias=True)
#         (to_out): ModuleList(
#           (0): Linear(in_features=1536, out_features=1536, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (attn2): Attention(
#         (norm_q): RMSNorm()
#         (norm_k): RMSNorm()
#         (to_q): Linear(in_features=1536, out_features=1536, bias=True)
#         (to_k): Linear(in_features=1536, out_features=1536, bias=True)
#         (to_v): Linear(in_features=1536, out_features=1536, bias=True)
#         (to_out): ModuleList(
#           (0): Linear(in_features=1536, out_features=1536, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (norm2): FP32LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
#       (ffn): FeedForward(
#         (net): ModuleList(
#           (0): GELU(
#             (proj): Linear(in_features=1536, out_features=8960, bias=True)
#           )
#           (1): Dropout(p=0.0, inplace=False)
#           (2): Linear(in_features=8960, out_features=1536, bias=True)
#         )
#       )
#       (norm3): FP32LayerNorm((1536,), eps=1e-06, elementwise_affine=False)
#     )
#   )
#   (norm_out): FP32LayerNorm((1536,), eps=1e-06, elementwise_affine=False)
#   (proj_out): Linear(in_features=1536, out_features=64, bias=True)
# )
