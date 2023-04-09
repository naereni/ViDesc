# -*- coding: utf-8 -*-
import pickle

import cv2
import pandas as pd
import PIL.Image
import torch
from config import Config
from sklearn.model_selection import train_test_split
from tqdm.contrib import tzip
from transformers import GPT2Tokenizer, XCLIPModel, XCLIPProcessor

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def read_video(
    config: Config, path: str, frames_num: int = 16
) -> list[PIL.Image]:
    frames: list[PIL.Image] = []
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


def extract_features(
    args: Config,
    encoder: XCLIPModel,
    processor: XCLIPProcessor,
    tokenizer: GPT2Tokenizer,
) -> None:
    df = pd.read_csv(args.dir + args.data_csv)[:50]
    train_df, test_df = train_test_split(df, test_size=0.1)
    train_encoder_embeds, train_captions_tokens = [], []
    test_encoder_embeds, test_captions_tokens = [], []
    encoder = encoder.to(DEVICE)
    for video_desc, video_name in tzip(
        train_df.caption, train_df.paths, desc="train_features"
    ):
        video_path = f"{args.dir}/{args.videos_path}/{video_name}"
        video = read_video(args, video_path)

        inputs = processor(videos=list(video), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embedding = encoder.get_video_features(**inputs).cpu().unsqueeze(0)
        train_encoder_embeds.append(embedding)

        video_desc = video_desc.replace("\n", "")
        caption = f"Описание видео: {video_desc}"
        train_captions_tokens.append(
            torch.tensor(tokenizer.encode(caption), dtype=torch.int64)
        )

    train_embeddings = {
        "encoder_embeds": torch.cat(train_encoder_embeds, dim=0),
        "captions_tokens": train_captions_tokens,
    }

    for video_desc, video_name in tzip(
        test_df.caption, test_df.paths, desc="test_features"
    ):
        video_path = f"{args.dir}/{args.videos_path}/{video_name}"
        video = read_video(args, video_path)

        inputs = processor(videos=list(video), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embedding = encoder.get_video_features(**inputs).cpu().unsqueeze(0)
        test_encoder_embeds.append(embedding)

        video_desc = video_desc.replace("\n", "")
        caption = f"Описание видео: {video_desc}"
        test_captions_tokens.append(
            torch.tensor(tokenizer.encode(caption), dtype=torch.int64)
        )

    test_embeddings = {
        "encoder_embeds": torch.cat(test_encoder_embeds, dim=0),
        "captions_tokens": test_captions_tokens,
    }

    print("%0d train embeddings saved " % len(train_encoder_embeds))
    print("%0d test embeddings saved " % len(test_encoder_embeds))

    with open(args.dir + args.train_features_path, "wb") as embed_dump:
        pickle.dump(train_embeddings, embed_dump)

    with open(args.dir + args.test_features_path, "wb") as embed_dump:
        pickle.dump(test_embeddings, embed_dump)
