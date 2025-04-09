# Running tests

TODO(aryan): everything here needs to be improved.

## `trainer/` fast tests

- For SFT tests: `test_sft_trainer.py`
- For Control tests: `test_control_trainer.py`

Accelerate:

```
# world_size=1 tests
accelerate launch --config_file accelerate_configs/uncompiled_1.yaml -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_1___batch_size_1 and ___Accelerate"
accelerate launch --config_file accelerate_configs/uncompiled_1.yaml -m pytest -s tests/trainer/test_sft_trainer.py -k "test___layerwise_upcasting___dp_degree_1___batch_size_1 and ___Accelerate"

# world_size=2 tests
accelerate launch --config_file accelerate_configs/uncompiled_2.yaml -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_2___batch_size_1 and ___Accelerate"
```

PTD:

```
# world_size=1 tests
torchrun --nnodes=1 --nproc_per_node 1 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_1___batch_size_1 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 1 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___layerwise_upcasting___dp_degree_1___batch_size_1 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 1 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_1___batch_size_2 and ___PTD"

# world_size=2 tests
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_2___batch_size_1 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___layerwise_upcasting___dp_degree_2___batch_size_1 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_2___batch_size_2 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_shards_2___batch_size_1 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_shards_2___batch_size_2 and ___PTD"
torchrun --nnodes=1 --nproc_per_node 2 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___tp_degree_2___batch_size_2 and ___PTD"

# world_size=4 tests
torchrun --nnodes=1 --nproc_per_node 4 -m pytest -s tests/trainer/test_sft_trainer.py -k "test___dp_degree_2___dp_shards_2___batch_size_1 and ___PTD"
```
