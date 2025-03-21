import pickle
from typing import Any, Dict

import torch.distributed.checkpoint.stateful
import torchdata.stateful_dataloader

from finetrainers.logging import get_logger


logger = get_logger()


class DPDataLoader(torchdata.stateful_dataloader.StatefulDataLoader, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(
        self,
        rank: int,
        dataset: torch.utils.data.IterableDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn=None,
    ) -> None:
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

        self._dp_rank = rank
        self._rank_id = f"dp_rank_{rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}")
            return

        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))
