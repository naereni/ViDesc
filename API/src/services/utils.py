import typing as tp

import cv2
import PIL
from PIL.Image import Image


class Config:
    def __init__(self) -> None:
        self.extract_size = 224, 224


def read_video(
    path: str,
    frames_num: int = 16,
    config: Config = Config(),
) -> tp.List[Image]:
    frames: tp.List[Image] = []
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length // (frames_num)
    current_frame = 0
    for i in range(length):
        ret, frame = cap.read(current_frame)
        if ret and i == current_frame and len(frames) < frames_num:
            frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.thumbnail(config.extract_size, PIL.Image.ANTIALIAS)
            frames.append(frame)
            current_frame += N
    cap.release()
    return frames
