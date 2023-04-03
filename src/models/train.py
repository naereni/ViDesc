# -*- coding: utf-8 -*-
import os
import sys

import torch
import wandb
from model import GPT2_Decoder
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModel,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)
from utils import extract_features

from datasets import ClipCocoDataset

wandb.init(project="ViDesc-training")

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class Config:
    def __init__(self) -> None:
        self.dir_path = "/content/videos_train"
        self.train_csv = "/content/ru_mixkit_train.csv"
        self.encoder_backbone = "microsoft/xclip-base-patch16-16-frames"
        self.decoder_backbone = "sberbank-ai/rugpt3small_based_on_gpt2"
        self.features_path = "mixkit_features_train.pkl"
        self.out_dir = "checkpoints"
        self.prefix = "ViDesc"
        self.epochs = 10
        self.extract_size = 224, 224
        self.save_every = 1
        self.prefix_length = 40
        self.prefix_size = 768
        self.bs = 10
        self.only_prefix = False
        self.lr = 5e-3
        self.warmup_steps = 5000


config = Config()


def train(
    config: Config,
    dataset: ClipCocoDataset,
    model: GPT2_Decoder,
) -> GPT2_Decoder:
    epochs = config.epochs
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    model = model.to(DEVICE)
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.995))

    train_dataloader = DataLoader(
        dataset, batch_size=config.bs, shuffle=True, drop_last=True
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=epochs * len(train_dataloader),
    )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=config.prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = (
                tokens.to(DEVICE),
                mask.to(DEVICE),
                prefix.to(DEVICE, dtype=torch.float32),
            )

            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1 : -1]

            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.flatten(),
                ignore_index=0,
            )
            loss.backward()

            optimizer.step()
            scheduler.step()
            progress.set_postfix({"loss": loss.item()})
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss.item()})

            progress.update()

            del tokens
            del mask
            del prefix
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()

        progress.close()
        if epoch % config.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    config.out_dir, f"{config.prefix}-{epoch:03d}.pt"
                ),
            )
    return model


if __name__ == "__main__":
    wandb.config = {
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "batch_size": config.bs,
    }

    print("--- Create models ---")
    encoder = AutoModel.from_pretrained(config.encoder_backbone)
    processor = AutoProcessor.from_pretrained(config.encoder_backbone)

    decoder = GPT2_Decoder(
        prefix_length=config.prefix_length,
        backbone=config.decoder_backbone,
        prefix_size=config.prefix_size,
    )

    if not os.path.exists(config.features_path):
        print("--- Feature extraction ---")
        extract_features(
            config, encoder=encoder, processor=processor, need_dump=True
        )

    print("--- Create dataset ---")
    dataset = ClipCocoDataset(
        data_path=config.features_path,
        prefix_length=config.prefix_length,
        gpt2_type=config.decoder_backbone,
    )

    print("--- Start training ---")
    train(
        config,
        dataset,
        decoder,
    )
