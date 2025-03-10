from typing import Any, Dict

import torch
import torch.distributed.checkpoint.stateful

from ...logging import get_logger
from ...processors import CannyProcessor
from .config import ControlType


logger = get_logger()


class IterableControlDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(self, dataset: torch.utils.data.IterableDataset, control_type: str):
        super().__init__()

        self.dataset = dataset
        self.control_type = control_type

        if control_type == ControlType.CANNY:
            self.control_processors = [
                CannyProcessor(["control_output"], input_names={"image": "input", "video": "input"})
            ]

        logger.info("Initialized IterableControlDataset with the following configuration")

    def __iter__(self):
        logger.info("Starting IterableControlDataset")

        for data in iter(self.dataset):
            control_augmented_data = self._run_control_processors(data)
            yield control_augmented_data

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)

    def state_dict(self):
        return self.dataset.state_dict()

    def _run_control_processors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.control_type == ControlType.CUSTOM:
            return data
        shallow_copy_data = dict(data.items())
        is_image_control = "image" in shallow_copy_data
        is_video_control = "video" in shallow_copy_data
        if (is_image_control + is_video_control) != 1:
            raise ValueError("Exactly one of 'image' or 'video' should be present in the data.")
        for processor in self.control_processors:
            result = processor(**shallow_copy_data)
            result_keys = set(result.keys())
            repeat_keys = result_keys.intersection(shallow_copy_data.keys())
            if repeat_keys:
                logger.warning(
                    f"Processor {processor.__class__.__name__} returned keys that already exist in "
                    f"conditions: {repeat_keys}. Overwriting the existing values, but this may not "
                    f"be intended. Please rename the keys in the processor to avoid conflicts."
                )
            shallow_copy_data.update(result)
        if "control_output" in shallow_copy_data:
            if is_image_control:
                shallow_copy_data["control_image"] = shallow_copy_data.pop("control_output")
            else:
                shallow_copy_data["control_video"] = shallow_copy_data.pop("control_output")
        return shallow_copy_data
