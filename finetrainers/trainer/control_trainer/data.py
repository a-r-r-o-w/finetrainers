from typing import Any, Dict

import torch
import torch.distributed.checkpoint.stateful
from diffusers.video_processor import VideoProcessor

from ... import functional as FF
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

        logger.info("Initialized IterableControlDataset")

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
        if "control_image" in data:
            if "image" in data:
                data["control_image"] = FF.resize_to_nearest_bucket_image(
                    data["control_image"], [data["image"].shape[-2:]], resize_mode="bicubic"
                )
            if "video" in data:
                batch_size, num_frames, num_channels, height, width = data["video"].shape
                data["control_video"], _first_frame_only = FF.resize_to_nearest_bucket_video(
                    data["control_video"], [[num_frames, height, width]], resize_mode="bicubic"
                )
                if _first_frame_only:
                    msg = (
                        "The number of frames in the control video is less than the minimum bucket size "
                        "specified. The first frame is being used as a single frame video. This "
                        "message is logged at the first occurence and for every 128th occurence "
                        "after that."
                    )
                    logger.log_freq("WARNING", "BUCKET_TEMPORAL_SIZE_UNAVAILABLE_CONTROL", msg, frequency=128)
                    data["control_video"] = data["control_video"][0]
            return data

        if "control_video" in data:
            if "image" in data:
                data["control_image"] = FF.resize_to_nearest_bucket_image(
                    data["control_video"][0], [data["image"].shape[-2:]], resize_mode="bicubic"
                )
            if "video" in data:
                batch_size, num_frames, num_channels, height, width = data["video"].shape
                data["control_video"], _first_frame_only = FF.resize_to_nearest_bucket_video(
                    data["control_video"], [[num_frames, height, width]], resize_mode="bicubic"
                )
                if _first_frame_only:
                    msg = (
                        "The number of frames in the control video is less than the minimum bucket size "
                        "specified. The first frame is being used as a single frame video. This "
                        "message is logged at the first occurence and for every 128th occurence "
                        "after that."
                    )
                    logger.log_freq("WARNING", "BUCKET_TEMPORAL_SIZE_UNAVAILABLE_CONTROL", msg, frequency=128)
                    data["control_video"] = data["control_video"][0]
            return data

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
            # Normalize to [-1, 1] range
            control_output = shallow_copy_data.pop("control_output")
            x_min = control_output.min()
            x_max = control_output.max()
            if not torch.isclose(x_min, x_max).any():
                control_output = 2 * (control_output - x_min) / (x_max - x_min) - 1
            key = "control_image" if is_image_control else "control_video"
            shallow_copy_data[key] = control_output
        return shallow_copy_data


class ValidationControlDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: torch.utils.data.IterableDataset, control_type: str):
        super().__init__()

        self.dataset = dataset
        self.control_type = control_type
        self._video_processor = VideoProcessor()

        if control_type == ControlType.CANNY:
            self.control_processors = [
                CannyProcessor(["control_output"], input_names={"image": "input", "video": "input"})
            ]

        logger.info("Initialized ValidationControlDataset")

    def __iter__(self):
        logger.info("Starting ValidationControlDataset")
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
        # These are already expected to be tensors
        if "control_image" in data or "control_video" in data:
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
            # Normalize to [-1, 1] range
            control_output = shallow_copy_data.pop("control_output")
            x_min = control_output.min()
            x_max = control_output.max()
            if not torch.isclose(x_min, x_max).any():
                control_output = 2 * (control_output - x_min) / (x_max - x_min) - 1
            key = "control_image" if is_image_control else "control_video"
            shallow_copy_data[key] = control_output
        return shallow_copy_data
