# Memory Optimization Techniques in Finetrainers

Finetrainers offers several techniques to optimize memory usage during training, allowing you to train models on hardware with less available GPU memory.

## Group Offloading

Group offloading is a memory optimization technique introduced in diffusers v0.33.0 that can significantly reduce GPU memory usage during training with minimal impact on training speed, especially when using CUDA devices that support streams.

Group offloading works by offloading groups of model layers to CPU when they're not needed and loading them back to GPU when they are. This is a middle ground between full model offloading (which keeps entire models on CPU) and sequential offloading (which keeps individual layers on CPU).

### Benefits of Group Offloading

- **Reduced Memory Usage**: Keep only parts of the model on GPU at any given time
- **Minimal Speed Impact**: When using CUDA streams, the performance impact is minimal
- **Configurable Balance**: Choose between block-level or leaf-level offloading based on your needs

### How to Enable Group Offloading

To enable group offloading, add the following flags to your training command:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

### Group Offloading Parameters

- `--enable_group_offload`: Enable group offloading (mutually exclusive with `--enable_model_cpu_offload`)
- `--group_offload_type`: Type of offloading to use
  - `block_level`: Offloads groups of layers based on blocks_per_group (default)
  - `leaf_level`: Offloads individual layers at the lowest level (similar to sequential offloading)
- `--group_offload_blocks_per_group`: Number of blocks per group when using `block_level` (default: 1)
- `--group_offload_use_stream`: Use CUDA streams for asynchronous data transfer (recommended for devices that support it)

### Example Usage

```bash
python train.py \
  --model_name flux \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --dataset_config "my_dataset_config.json" \
  --output_dir "output_flux_lora" \
  --training_type lora \
  --train_steps 5000 \
  --enable_group_offload \
  --group_offload_type block_level \
  --group_offload_blocks_per_group 1 \
  --group_offload_use_stream
```

### Memory-Performance Tradeoffs

- For maximum memory savings with slower performance: Use `--group_offload_type leaf_level`
- For balanced memory savings with better performance: Use `--group_offload_type block_level` with `--group_offload_blocks_per_group 1` and `--group_offload_use_stream`
- For minimal memory savings but best performance: Increase `--group_offload_blocks_per_group` to a higher value

> **Note**: Group offloading requires diffusers v0.33.0 or higher.

## Other Memory Optimization Techniques

Finetrainers also supports other memory optimization techniques that can be used independently or in combination:

- **Model CPU Offloading**: `--enable_model_cpu_offload` (mutually exclusive with group offloading)
- **Gradient Checkpointing**: `--gradient_checkpointing`
- **Layerwise Upcasting**: Using low precision (e.g., FP8) for storage with higher precision for computation
- **VAE Optimizations**: `--enable_slicing` and `--enable_tiling`
- **Precomputation**: `--enable_precomputation` to precompute embeddings

Combining these techniques can significantly reduce memory requirements for training large models.