import typing as tp

import cv2
import PIL
from PIL.Image import Image


def read_video(
    config: tp.Dict, path: str, frames_num: int = 16
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
