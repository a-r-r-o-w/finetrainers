# Running tests

TODO(aryan): everything here needs to be improved.

## `trainer/` fast tests

```
# world_size=1 tests
torchrun --nnodes=1 --nproc_per_node 1 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_degree_1___batch_size_1
torchrun --nnodes=1 --nproc_per_node 1 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_degree_1___batch_size_2

# world_size=2 tests
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_degree_2___batch_size_1
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_degree_2___batch_size_2
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_shards_2___batch_size_1
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_shards_2___batch_size_2
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k test___tp_degree_2___batch_size_2

# world_size=4 tests
torchrun --nnodes=1 --nproc_per_node 4 -m pytest -s tests/trainer/test_sft_trainer.py -k test___dp_degree_2___dp_shards_2___batch_size_1
```
