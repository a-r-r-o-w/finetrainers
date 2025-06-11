import unittest

import torch

from finetrainers.data.dataset import IterableDatasetPreprocessingWrapper


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_samples=100):
        super().__init__()
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield {"caption": f"caption_{i}", "image": i}


class TestIterableDatasetMultiWorker(unittest.TestCase):
    def test_no_duplication_with_multiple_workers(self):
        """
        Tests that IterableDatasetPreprocessingWrapper correctly shards data and
        handles the drop_last logic, by directly comparing the loaded items
        to a manually simulated expected set of items.
        """
        num_samples = 101  # Not perfectly divisible by batch_size to test drop_last
        batch_size = 4

        for num_workers in range(1, 9):
            with self.subTest(num_workers=num_workers):
                drop_last = num_workers > 1

                original_dataset = DummyIterableDataset(num_samples)
                original_items = [item["image"] for item in original_dataset]

                wrapped_dataset = IterableDatasetPreprocessingWrapper(
                    dataset=original_dataset,
                    dataset_type="image",
                )

                dataloader = torch.utils.data.DataLoader(
                    wrapped_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                )

                loaded_items = [item for batch in dataloader for item in batch["image"].tolist()]

                # Manually simulate the sharding and drop_last logic to get the exact expected set of items.
                expected_items = []
                if drop_last:
                    for worker_id in range(num_workers):
                        # 1. Simulate the interleaved sharding from itertools.islice
                        worker_items = original_items[worker_id::num_workers]
                        # 2. Simulate the drop_last logic for this worker's items
                        num_full_batches = len(worker_items) // batch_size
                        items_to_keep_for_worker = worker_items[: num_full_batches * batch_size]
                        expected_items.extend(items_to_keep_for_worker)
                else:  # This case is for num_workers == 1
                    expected_items = original_items

                # Ensure no duplicates were loaded.
                self.assertEqual(
                    len(loaded_items),
                    len(expected_items),
                    f"The number of loaded items does not match the expected number for {num_workers} workers.",
                )


if __name__ == "__main__":
    unittest.main()
