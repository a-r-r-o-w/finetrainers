import random
from typing import Any, Dict, Optional

import torch
import torch.distributed.checkpoint.stateful
from diffusers.video_processor import VideoProcessor

import finetrainers.functional as FF
from finetrainers.logging import get_logger
from finetrainers.processors import CannyProcessor, CopyProcessor

from .config import ControlType, FrameConditioningType


logger = get_logger()


class IterableControlDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    def __init__(
        self, dataset: torch.utils.data.IterableDataset, control_type: str, device: Optional[torch.device] = None
    ):
        super().__init__()

        self.dataset = dataset
        self.control_type = control_type

        self.control_processors = []
        if control_type == ControlType.CANNY:
            self.control_processors.append(
                CannyProcessor(
                    output_names=["control_output"], input_names={"image": "input", "video": "input"}, device=device
                )
            )
        elif control_type == ControlType.NONE:
            self.control_processors.append(
                CopyProcessor(output_names=["control_output"], input_names={"image": "input", "video": "input"})
            )

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
            # TODO(aryan): need to specify a dim for normalize here across channels
            control_output = FF.normalize(control_output, min=-1.0, max=1.0)
            key = "control_image" if is_image_control else "control_video"
            shallow_copy_data[key] = control_output
        return shallow_copy_data


class ValidationControlDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, dataset: torch.utils.data.IterableDataset, control_type: str, device: Optional[torch.device] = None
    ):
        super().__init__()

        self.dataset = dataset
        self.control_type = control_type
        self.device = device
        self._video_processor = VideoProcessor()

        self.control_processors = []
        if control_type == ControlType.CANNY:
            self.control_processors.append(
                CannyProcessor(["control_output"], input_names={"image": "input", "video": "input"}, device=device)
            )
        elif control_type == ControlType.NONE:
            self.control_processors.append(
                CopyProcessor(["control_output"], input_names={"image": "input", "video": "input"})
            )

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
            if torch.is_tensor(control_output):
                # TODO(aryan): need to specify a dim for normalize here across channels
                control_output = FF.normalize(control_output, min=-1.0, max=1.0)
                ndim = control_output.ndim
                assert 3 <= ndim <= 5, "Control output should be at least ndim=3 and less than or equal to ndim=5"
                if ndim == 5:
                    control_output = self._video_processor.postprocess_video(control_output, output_type="pil")
                else:
                    if ndim == 3:
                        control_output = control_output.unsqueeze(0)
                    control_output = self._video_processor.postprocess(control_output, output_type="pil")[0]
            key = "control_image" if is_image_control else "control_video"
            shallow_copy_data[key] = control_output
        return shallow_copy_data


# TODO(aryan): write a test for this function
def apply_frame_conditioning_on_latents(
    latents: torch.Tensor,
    expected_num_frames: int,
    channel_dim: int,
    frame_dim: int,
    frame_conditioning_type: FrameConditioningType,
    frame_conditioning_index: Optional[int] = None,
    concatenate_mask: bool = False,
) -> torch.Tensor:
    num_frames = latents.size(frame_dim)
    mask = torch.zeros_like(latents)

    if frame_conditioning_type == FrameConditioningType.INDEX:
        frame_index = min(frame_conditioning_index, num_frames - 1)
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = frame_index
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.PREFIX:
        frame_index = random.randint(1, num_frames)
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = slice(0, frame_index)  # Keep frames 0 to frame_index-1
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.RANDOM:
        # Zero or more random frames to keep
        num_frames_to_keep = random.randint(1, num_frames)
        frame_indices = random.sample(range(num_frames), num_frames_to_keep)
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = frame_indices
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.FIRST_AND_LAST:
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = 0
        mask[tuple(indexing)] = 1
        indexing[frame_dim] = num_frames - 1
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.FULL:
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = slice(0, num_frames)
        mask[tuple(indexing)] = 1

    if latents.size(frame_dim) >= expected_num_frames:
        slicing = [slice(None)] * latents.ndim
        slicing[frame_dim] = slice(expected_num_frames)
        latents = latents[tuple(slicing)]
        mask = mask[tuple(slicing)]
    else:
        pad_size = expected_num_frames - num_frames
        pad_shape = list(latents.shape)
        pad_shape[frame_dim] = pad_size
        padding = latents.new_zeros(pad_shape)
        latents = torch.cat([latents, padding], dim=frame_dim)
        mask = torch.cat([mask, padding], dim=frame_dim)

    if concatenate_mask:
        slicing = [slice(None)] * latents.ndim
        slicing[channel_dim] = 0
        latents = torch.cat([latents, mask], dim=channel_dim)

    return latents
