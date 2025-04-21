import copy

import torch
import torch.distributed as dist
from diffusers import LTXVideoTransformer3DModel
from torch._utils import _get_device_module
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    RowwiseParallel,
)


# from torch.utils._python_dispatch import TorchDispatchMode


DEVICE_TYPE = "cuda"
PG_BACKEND = "nccl"
DEVICE_COUNT = _get_device_module(DEVICE_TYPE).device_count()


def main(world_size: int, rank: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(rank)

    CHANNELS = 128
    CROSS_ATTENTION_DIM = 2048
    CAPTION_CHANNELS = 4096
    NUM_LAYERS = 28
    NUM_ATTENTION_HEADS = 32
    ATTENTION_HEAD_DIM = 64

    # CHANNELS = 4
    # CROSS_ATTENTION_DIM = 32
    # CAPTION_CHANNELS = 64
    # NUM_LAYERS = 1
    # NUM_ATTENTION_HEADS = 4
    # ATTENTION_HEAD_DIM = 8

    config = {
        "in_channels": CHANNELS,
        "out_channels": CHANNELS,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "attention_head_dim": ATTENTION_HEAD_DIM,
        "cross_attention_dim": CROSS_ATTENTION_DIM,
        "num_layers": NUM_LAYERS,
        "activation_fn": "gelu-approximate",
        "qk_norm": "rms_norm_across_heads",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-6,
        "caption_channels": CAPTION_CHANNELS,
        "attention_bias": True,
        "attention_out_bias": True,
    }

    # Normal model
    torch.manual_seed(0)
    model = LTXVideoTransformer3DModel(**config).to(DEVICE_TYPE)

    # TP model
    model_tp = copy.deepcopy(model)
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    print(f"Device mesh: {device_mesh}")

    transformer_tp_plan = {
        # ===== Condition embeddings =====
        # "time_embed.emb.timestep_embedder.linear_1": ColwiseParallel(),
        # "time_embed.emb.timestep_embedder.linear_2": RowwiseParallel(output_layouts=Shard(-1)),
        # "time_embed.linear": ColwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
        # "time_embed": PrepareModuleOutput(output_layouts=(Replicate(), Shard(-1)), desired_output_layouts=(Replicate(), Replicate())),
        # "caption_projection.linear_1": ColwiseParallel(),
        # "caption_projection.linear_2": RowwiseParallel(),
        # "rope": PrepareModuleOutput(output_layouts=(Replicate(), Replicate()), desired_output_layouts=(Shard(1), Shard(1)), use_local_output=False),
        # ===== =====
    }

    for block in model_tp.transformer_blocks:
        block_tp_plan = {}

        # ===== Attention =====
        # 8 all-to-all, 3 all-reduce
        # block_tp_plan["attn1.to_q"] = ColwiseParallel(use_local_output=False)
        # block_tp_plan["attn1.to_k"] = ColwiseParallel(use_local_output=False)
        # block_tp_plan["attn1.to_v"] = ColwiseParallel(use_local_output=False)
        # block_tp_plan["attn1.norm_q"] = SequenceParallel()
        # block_tp_plan["attn1.norm_k"] = SequenceParallel()
        # block_tp_plan["attn1.to_out.0"] = RowwiseParallel(input_layouts=Shard(1))
        # block_tp_plan["attn2.to_q"] = ColwiseParallel(use_local_output=False)
        # block_tp_plan["attn2.to_k"] = ColwiseParallel(use_local_output=False)
        # block_tp_plan["attn2.to_v"] = ColwiseParallel(use_local_output=False)
        # block_tp_plan["attn2.norm_q"] = SequenceParallel()
        # block_tp_plan["attn2.norm_k"] = SequenceParallel()
        # block_tp_plan["attn2.to_out.0"] = RowwiseParallel(input_layouts=Shard(1))
        # ===== =====

        block_tp_plan["ff.net.0.proj"] = ColwiseParallel()
        block_tp_plan["ff.net.2"] = RowwiseParallel()
        parallelize_module(block, device_mesh, block_tp_plan)

    parallelize_module(model_tp, device_mesh, transformer_tp_plan)

    comm_mode = CommDebugMode()

    batch_size = 2
    num_frames, height, width = 49, 512, 512
    temporal_compression_ratio, spatial_compression_ratio = 8, 32
    latent_num_frames, latent_height, latent_width = (
        (num_frames - 1) // temporal_compression_ratio + 1,
        height // spatial_compression_ratio,
        width // spatial_compression_ratio,
    )
    video_sequence_length = latent_num_frames * latent_height * latent_width
    caption_sequence_length = 64

    hidden_states = torch.randn(batch_size, video_sequence_length, CHANNELS, device=DEVICE_TYPE)
    encoder_hidden_states = torch.randn(batch_size, caption_sequence_length, CAPTION_CHANNELS, device=DEVICE_TYPE)
    encoder_attention_mask = None
    timestep = torch.randint(0, 1000, (batch_size, 1), device=DEVICE_TYPE)
    inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "timestep": timestep,
        "num_frames": latent_num_frames,
        "height": latent_height,
        "width": latent_width,
        "rope_interpolation_scale": [1 / (8 / 25), 8, 8],
        "return_dict": False,
    }

    output = model(**inputs)[0]

    with comm_mode:
        output_tp = model_tp(**inputs)[0]

    output_tp = (
        output_tp.redistribute(output_tp.device_mesh, [Replicate()]).to_local()
        if isinstance(output_tp, DTensor)
        else output_tp
    )

    print("Output shapes:", output.shape, output_tp.shape)
    print(
        "Comparing output:",
        rank,
        torch.allclose(output, output_tp, atol=1e-5, rtol=1e-5),
        (output - output_tp).abs().max(),
    )
    print(f"Max memory reserved ({rank=}): {torch.cuda.max_memory_reserved(rank) / 1024**3:.2f} GB")

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


dist.init_process_group(PG_BACKEND)
WORLD_SIZE = dist.get_world_size()
RANK = dist.get_rank()

torch.cuda.set_device(RANK)

if RANK == 0:
    print(f"World size: {WORLD_SIZE}")
    print(f"Device count: {DEVICE_COUNT}")

try:
    with torch.no_grad():
        main(WORLD_SIZE, RANK)
finally:
    dist.destroy_process_group()


# LTXVideoTransformer3DModel(
#   (proj_in): Linear(in_features=128, out_features=2048, bias=True)
#   (time_embed): AdaLayerNormSingle(
#     (emb): PixArtAlphaCombinedTimestepSizeEmbeddings(
#       (time_proj): Timesteps()
#       (timestep_embedder): TimestepEmbedding(
#         (linear_1): Linear(in_features=256, out_features=2048, bias=True)
#         (act): SiLU()
#         (linear_2): Linear(in_features=2048, out_features=2048, bias=True)
#       )
#     )
#     (silu): SiLU()
#     (linear): Linear(in_features=2048, out_features=12288, bias=True)
#   )
#   (caption_projection): PixArtAlphaTextProjection(
#     (linear_1): Linear(in_features=4096, out_features=2048, bias=True)
#     (act_1): GELU(approximate='tanh')
#     (linear_2): Linear(in_features=2048, out_features=2048, bias=True)
#   )
#   (rope): LTXVideoRotaryPosEmbed()
#   (transformer_blocks): ModuleList(
#     (0-27): 28 x LTXVideoTransformerBlock(
#       (norm1): RMSNorm()
#       (attn1): Attention(
#         (norm_q): RMSNorm()
#         (norm_k): RMSNorm()
#         (to_q): Linear(in_features=2048, out_features=2048, bias=True)
#         (to_k): Linear(in_features=2048, out_features=2048, bias=True)
#         (to_v): Linear(in_features=2048, out_features=2048, bias=True)
#         (to_out): ModuleList(
#           (0): Linear(in_features=2048, out_features=2048, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (norm2): RMSNorm()
#       (attn2): Attention(
#         (norm_q): RMSNorm()
#         (norm_k): RMSNorm()
#         (to_q): Linear(in_features=2048, out_features=2048, bias=True)
#         (to_k): Linear(in_features=2048, out_features=2048, bias=True)
#         (to_v): Linear(in_features=2048, out_features=2048, bias=True)
#         (to_out): ModuleList(
#           (0): Linear(in_features=2048, out_features=2048, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (ff): FeedForward(
#         (net): ModuleList(
#           (0): GELU(
#             (proj): Linear(in_features=2048, out_features=8192, bias=True)
#           )
#           (1): Dropout(p=0.0, inplace=False)
#           (2): Linear(in_features=8192, out_features=2048, bias=True)
#         )
#       )
#     )
#   )
#   (norm_out): LayerNorm((2048,), eps=1e-06, elementwise_affine=False)
#   (proj_out): Linear(in_features=2048, out_features=128, bias=True)
# )
