# ===== THIS FILE ONLY EXISTS FOR THE TIME BEING SINCE I DID NOT KNOW WHERE TO PUT IT =====

from dataclasses import dataclass
from typing import Any, List

from PIL.Image import Image


@dataclass
class Artifact:
    type: str
    value: Any
    file_extension: str


@dataclass
class ImageArtifact(Artifact):
    value: Image

    def __init__(self, value: Image):
        super().__init__(type="image", value=value, file_extension="png")


@dataclass
class VideoArtifact(Artifact):
    value: List[Image]

    def __init__(self, value: List[Image]):
        super().__init__(type="video", value=value, file_extension="mp4")
