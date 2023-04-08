# -*- coding: utf-8 -*-
import os
import sys

# import wandb
import torch
from config import Config
from model import GPT2_Decoder
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    GPT2Tokenizer,
    XCLIPModel,
    XCLIPProcessor,
    get_linear_schedule_with_warmup,
)
from utils import extract_features

from datasets import ClipCocoDataset

# wandb.init(project="ViDesc-training")

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

config = Config()


def train(
    config: Config,
    dataset: ClipCocoDataset,
    model: GPT2_Decoder,
) -> GPT2_Decoder:
    epochs = config.epochs
    if not os.path.exists(config.dir + config.out_dir):
        os.makedirs(config.dir + config.out_dir)

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

            # wandb.log({"loss": loss.item()})

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
    # wandb.config = {
    #     "learning_rate": config.lr,
    #     "epochs": config.epochs,
    #     "batch_size": config.bs,
    # }

    print("--- Create models ---")
    encoder = XCLIPModel.from_pretrained(config.encoder_backbone)
    processor = XCLIPProcessor.from_pretrained(config.encoder_backbone)

    decoder = GPT2_Decoder(
        prefix_length=config.prefix_length,
        backbone=config.decoder_backbone,
        prefix_size=config.prefix_size,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(config.decoder_backbone)

    if not os.path.exists(config.dir + config.train_features_path):
        print("--- Feature extraction ---")
        extract_features(config, encoder, processor, tokenizer)

    print("--- Create dataset ---")
    train_dataset = ClipCocoDataset(
        data_path=config.dir + config.train_features_path,
        prefix_length=config.prefix_length,
        tokenizer_type=config.decoder_backbone,
    )

    test_dataset = ClipCocoDataset(
        data_path=config.dir + config.test_features_path,
        prefix_length=config.prefix_length,
        tokenizer_type=config.decoder_backbone,
    )

    print("--- Start training ---")
    sys.stdout.flush()
    train(
        dataset=train_dataset,
        config=config,
        model=decoder,
    )
