import pathlib
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from tqdm.auto import tqdm

from .. import utils


class DistributedDataPreprocessor:
    def __init__(
        self,
        rank: int,
        num_items: int,
        processor_fn: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        save_dir: str,
    ) -> None:
        self._rank = rank
        self._num_items = num_items
        self._processor_fn = processor_fn
        self._save_dir = pathlib.Path(save_dir)

        self._cached_samples = []
        self._preprocessed_iterator: "PreprocessedDataIterable" = None

        self._save_dir.mkdir(parents=True, exist_ok=True)

        subdirectories = [f for f in self._save_dir.iterdir() if f.is_dir()]
        utils.delete_files(subdirectories)

    def consume(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        for i in tqdm(range(self._num_items), desc=f"Rank {self._rank}", total=self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)
            item = self._processor_fn[data_type](**item, **components, generator=generator)
            _save_item(self._rank, i, item, self._save_dir, data_type)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []
            utils.free_memory()

        self._preprocessed_iterator = PreprocessedDataIterable(self._rank, self._save_dir, data_type)
        return iter(self._preprocessed_iterator)

    def consume_once(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        for i in tqdm(range(self._num_items), desc=f"Processing data on rank {self._rank}", total=self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)
            item = self._processor_fn[data_type](**item, **components, generator=generator)
            _save_item(self._rank, i, item, self._save_dir, data_type)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []
            utils.free_memory()

        self._preprocessed_iterator = PreprocessedOnceDataIterable(self._rank, self._save_dir, data_type)
        return iter(self._preprocessed_iterator)

    @property
    def requires_data(self):
        if self._preprocessed_iterator is None:
            return True
        return self._preprocessed_iterator.requires_data


class PreprocessedDataIterable:
    def __init__(self, rank: int, save_dir: str, data_type: str) -> None:
        self._rank = rank
        self._save_dir = pathlib.Path(save_dir)
        self._num_items = len(list(self._save_dir.glob(f"{data_type}-{rank}-*.pt")))
        self._data_type = data_type

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for i in range(self._num_items):
            if i == self._num_items - 1:
                self._requires_data = True
            yield _load_item(self._rank, i, self._save_dir, self._data_type)

    def __len__(self) -> int:
        return self._num_items

    @property
    def requires_data(self):
        return self._requires_data


class PreprocessedOnceDataIterable:
    def __init__(self, rank: int, save_dir: str, data_type: str) -> None:
        self._rank = rank
        self._save_dir = pathlib.Path(save_dir)
        self._num_items = len(list(self._save_dir.glob(f"{data_type}-{rank}-*.pt")))
        self._data_type = data_type

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        index = 0
        while True:
            yield _load_item(self._rank, index, self._save_dir, self._data_type)
            index = (index + 1) % self._num_items

    def __len__(self) -> int:
        return self._num_items

    @property
    def requires_data(self):
        return self._requires_data


def _save_item(rank: int, index: int, item: Dict[str, Any], directory: pathlib.Path, data_type: str) -> None:
    filename = directory / f"{data_type}-{rank}-{index}.pt"
    torch.save(item, filename.as_posix())


def _load_item(rank: int, index: int, directory: pathlib.Path, data_type: str) -> Dict[str, Any]:
    filename = directory / f"{data_type}-{rank}-{index}.pt"
    return torch.load(filename.as_posix(), weights_only=True)
