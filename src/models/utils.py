import pickle

import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def read_video(path: str, frames_num: int = 16) -> list[Image]:
    frames: list[Image] = []
    cap = cv2.VideoCapture(path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length // (frames_num)

    current_frame = 0
    for i in range(length):
        ret, frame = cap.read(current_frame)

        if ret and i == current_frame and len(frames) < frames_num:
            size = 224, 224
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.thumbnail(size, Image.ANTIALIAS)

            frames.append(frame)
            current_frame += N

    cap.release()
    return frames


def generate_embeddings(
    df: pd.DataFrame,
    dir_path: str,
    dump_path: str,
    encoder: AutoModel,
    processor: AutoProcessor,
    need_dump: bool = False,
) -> dict[torch.Tensor, list[str]]:
    encoder_embeds, all_captions = [], []
    encoder = encoder.to(DEVICE)

    for video_desc, video_name in zip(df.captions, df.paths):
        video_path = f"{dir_path}/{video_name[0]}.mp4"
        video = read_video(video_path)
        inputs = processor(videos=list(video), return_tensors="pt")
        inputs = inputs.pixel_values.squeeze(0).to(DEVICE)
        text = f"Caption: {video_desc}<|endoftext|>"
        with torch.no_grad():
            embedding = (
                encoder.vision_model(inputs).pooler_output.cpu().unsqueeze(0)
            )
        encoder_embeds.append(embedding)
        all_captions.append(text)

    embeddings = {
        "encoder_embeds": torch.cat(encoder_embeds, dim=0),
        "captions": all_captions,
    }

    if need_dump:
        with open(dump_path, "wb") as embed_dump:
            pickle.dump(embeddings, embed_dump)
    return embeddings
