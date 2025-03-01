from typing import Any, Dict, List, Tuple

import torch


class ResolutionSampler:
    def __init__(self, batch_size: int = 1, dim_keys: Dict[str, Tuple[int, ...]] = None) -> None:
        self.batch_size = batch_size
        self.dim_keys = dim_keys
        assert dim_keys is not None, "dim_keys must be provided"

        self._chosen_leader_key = None
        self._unsatisfied_buckets: Dict[Tuple[int, ...], List[Dict[Any, Any]]] = {}
        self._satisfied_buckets: List[Dict[Any, Any]] = []

    def consume(self, *dict_items: Dict[Any, Any]) -> None:
        if self._chosen_leader_key is None:
            self._determine_leader_item(*dict_items)
        self._update_buckets(*dict_items)

    def get_batch(self) -> List[Dict[str, Any]]:
        return list(zip(*self._satisfied_buckets.pop(-1)))

    @property
    def is_ready(self) -> bool:
        return len(self._satisfied_buckets) > 0

    def _determine_leader_item(self, *dict_items: Dict[Any, Any]) -> None:
        num_observed = 0
        for dict_item in dict_items:
            for key in self.dim_keys.keys():
                if key in dict_item.keys():
                    self._chosen_leader_key = key
                    if not torch.is_tensor(dict_item[key]):
                        raise ValueError(f"Leader key {key} must be a tensor")
                    num_observed += 1
        if num_observed > 1:
            raise ValueError(
                f"Only one leader key is allowed in provided list of data dictionaries. Found {num_observed} leader keys"
            )
        if self._chosen_leader_key is None:
            raise ValueError("No leader key found in provided list of data dictionaries")

    def _update_buckets(self, *dict_items: Dict[Any, Any]) -> None:
        chosen_value = [
            dict_item[self._chosen_leader_key]
            for dict_item in dict_items
            if self._chosen_leader_key in dict_item.keys()
        ]
        if len(chosen_value) == 0:
            raise ValueError(f"Leader key {self._chosen_leader_key} not found in provided list of data dictionaries")
        chosen_value = chosen_value[0]
        dims = tuple(chosen_value.size(x) for x in self.dim_keys[self._chosen_leader_key])
        if dims not in self._unsatisfied_buckets:
            self._unsatisfied_buckets[dims] = []
        self._unsatisfied_buckets[dims].append(dict_items)
        if len(self._unsatisfied_buckets[dims]) == self.batch_size:
            self._satisfied_buckets.append(self._unsatisfied_buckets.pop(dims))
