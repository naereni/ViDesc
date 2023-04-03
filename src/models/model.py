# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class MLP(nn.Module):
    def __init__(
        self,
        sizes: Tuple[int, ...],
        bias: bool = True,
        act: torch.nn = nn.Tanh,
    ) -> None:
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GPT2_Decoder(nn.Module):
    def __init__(
        self, backbone: str, prefix_length: int, prefix_size: int
    ) -> None:
        super(GPT2_Decoder, self).__init__()

        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(backbone)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP(
            (
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            )
        )

    def get_dummy_token(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(
            inputs_embeds=embedding_cat, labels=labels, attention_mask=mask
        )
        return out
