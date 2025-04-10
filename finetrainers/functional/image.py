from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F


def center_crop_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    num_channels, height, width = image.shape
    crop_h, crop_w = size
    if height < crop_h or width < crop_w:
        raise ValueError(f"Image size {(height, width)} is smaller than the target size {size}.")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[:, top : top + crop_h, left : left + crop_w]


def resize_crop_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    num_channels, height, width = image.shape
    target_h, target_w = size
    scale = max(target_h / height, target_w / width)
    new_h, new_w = int(height * scale), int(width * scale)
    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return center_crop_image(image, size)


def bicubic_resize_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(image.unsqueeze(0), size=size, mode="bicubic", align_corners=False)[0]


def find_nearest_resolution_image(image: torch.Tensor, resolution_buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    num_channels, height, width = image.shape
    aspect_ratio = width / height

    def aspect_ratio_diff(bucket):
        return abs((bucket[1] / bucket[0]) - aspect_ratio), (-bucket[0], -bucket[1])

    return min(resolution_buckets, key=aspect_ratio_diff)


def resize_to_nearest_bucket_image(
    image: torch.Tensor,
    resolution_buckets: List[Tuple[int, int]],
    resize_mode: Literal["center_crop", "resize_crop", "bicubic"] = "bicubic",
) -> torch.Tensor:
    target_size = find_nearest_resolution_image(image, resolution_buckets)

    if resize_mode == "center_crop":
        return center_crop_image(image, target_size)
    elif resize_mode == "resize_crop":
        return resize_crop_image(image, target_size)
    elif resize_mode == "bicubic":
        return bicubic_resize_image(image, target_size)
    else:
        raise ValueError(
            f"Invalid resize_mode: {resize_mode}. Choose from 'center_crop', 'resize_crop', or 'bicubic'."
        )
