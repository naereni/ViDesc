# -*- coding: utf-8 -*-

import pickle
import sys
from typing import Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer


class ClipCocoDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prefix_length: int,
        gpt2_type: str,
        normalize_prefix: bool = False,
    ) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        with open(data_path, "rb") as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["encoder_embeds"]))

        sys.stdout.flush()
        self.prefixes = all_data["encoder_embeds"]
        self.captions = all_data["captions"]

        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        for i, caption in enumerate(tqdm(self.captions)):
            self.captions_tokens.append(
                torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
            )
            self.caption2embedding.append(self.prefixes[i])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

        all_len = torch.tensor(
            [len(self.captions_tokens[i]) for i in range(len(self))]
        ).float()
        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max())
        )

    def pad_tokens(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.zeros(padding, dtype=torch.int64) - 1)
            )
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(
        self, item: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[item]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix
