from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F


def center_crop_video(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    num_frames, num_channels, height, width = video.shape
    crop_h, crop_w = size
    if height < crop_h or width < crop_w:
        raise ValueError(f"Video size {(height, width)} is smaller than the target size {size}.")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return video[:, :, top : top + crop_h, left : left + crop_w]


def resize_crop_video(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    num_frames, num_channels, height, width = video.shape
    target_h, target_w = size
    scale = max(target_h / height, target_w / width)
    new_h, new_w = int(height * scale), int(width * scale)
    video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return center_crop_video(video, size)


def bicubic_resize_video(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    num_frames, num_channels, height, width = video.shape
    video = F.interpolate(video, size=size, mode="bicubic", align_corners=False)
    return video


def find_nearest_video_resolution(
    video: torch.Tensor, resolution_buckets: List[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    num_frames, num_channels, height, width = video.shape
    aspect_ratio = width / height
    possible_buckets = [b for b in resolution_buckets if b[0] <= num_frames]

    if not possible_buckets:
        best_frame_match = min(resolution_buckets, key=lambda b: abs(b[0] - num_frames))
    else:
        best_frame_match = max(possible_buckets, key=lambda b: b[0])

    frame_filtered_buckets = [b for b in resolution_buckets if b[0] == best_frame_match[0]]

    def aspect_ratio_diff(bucket):
        return abs((bucket[2] / bucket[1]) - aspect_ratio), (-bucket[1], -bucket[2])

    return min(frame_filtered_buckets, key=aspect_ratio_diff)


def resize_to_nearest_bucket_video(
    video: torch.Tensor,
    resolution_buckets: List[Tuple[int, int, int]],
    resize_mode: Literal["center_crop", "resize_crop", "bicubic"] = "bicubic",
) -> torch.Tensor:
    """
    Resizes a video tensor to the nearest resolution bucket using the specified mode.
    - It first finds a frame match with <= T frames.
    - Then, it selects the closest height/width bucket.

    Args:
        video (`torch.Tensor`):
            Input video tensor of shape `(B, T, C, H, W)`.
        resolution_buckets (`List[Tuple[int, int, int]]`):
            Available (num_frames, height, width) resolution buckets.
        resize_mode (`str`):
            One of ["center_crop", "resize_crop", "bicubic"].

    Returns:
        `torch.Tensor`:
            Resized video tensor of the nearest bucket resolution.
    """
    target_frames, target_h, target_w = find_nearest_video_resolution(video, resolution_buckets)

    # Adjust frame count: only interpolate frames if no lesser/equal frame count exists
    num_frames, num_channels, height, width = video.shape
    _first_frame_only = False
    if num_frames > target_frames:
        # Downsample: Select frames evenly
        indices = torch.linspace(0, num_frames - 1, target_frames).long()
        video = video[indices, :, :, :]
    elif num_frames < target_frames:
        _first_frame_only = False

    # Resize spatial resolution
    if resize_mode == "center_crop":
        return center_crop_video(video, (target_h, target_w)), _first_frame_only
    elif resize_mode == "resize_crop":
        return resize_crop_video(video, (target_h, target_w)), _first_frame_only
    elif resize_mode == "bicubic":
        return bicubic_resize_video(video, (target_h, target_w)), _first_frame_only
    else:
        raise ValueError(
            f"Invalid resize_mode: {resize_mode}. Choose from 'center_crop', 'resize_crop', or 'bicubic'."
        )
