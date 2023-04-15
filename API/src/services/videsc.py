from typing import Dict, List

from PIL.Image import Image
from src.services.video_descriptor import VideoDescription


class ViDesc:
    def __init__(self, descriptor: VideoDescription):
        self._descriptor = descriptor

    def predict(self, video: List[Image]) -> Dict[str, str]:
        return self._descriptor.get_caption(video)
