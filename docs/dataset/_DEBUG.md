# Distributed dataset debugging

>![NOTE]
> This doc page is intended for developers and contributors.

If the number of samples in the dataset is lower than the number of processes per node, the training will hand indefinitely. I haven't been able to pin down on how this could be fixed due to limited time, but basically:
- Start training with `--dp_degree 2` and `torchrun --standalone --nnodes=1 --nproc_per_node=2`. This launches training with DDP across 2 ranks.
- The dataset has `< dp_degree` samples
- When `datasets.distributed.split_dataset_by_node` is called, the data is distributed correctly to one rank, but the other rank hangs indefinitely. Due to this edge case, fast tests seem to fail.
- For now, we should just use `>= dp_degree` samples in the test dataset. However, should be fixed in the future.

Minimal reproducer:

```python
import torch
import torch.distributed as dist
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

ds = Dataset.from_dict({"x": [1]}).to_iterable_dataset()

dist.init_process_group()
rank, world_size = dist.get_rank(), dist.get_world_size()
ds = split_dataset_by_node(ds, rank=rank,world_size=world_size)
dl = DataLoader(ds)

exhausted = torch.zeros(world_size, dtype=torch.bool)

def loop():
    while True:
        print(rank, "hello", flush=True)
        yield from dl
        yield "end"

for x in loop():
    if x == "end":
        exhausted[rank] = True
        continue
    dist.all_reduce(exhausted)
    if torch.all(exhausted):
        break
    print(f"{rank} {x}", flush=True)
```
