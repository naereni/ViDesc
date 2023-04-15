import typing as tp

import torch
from PIL.Image import Image
from src.services.model import GPT2_Decoder
from torch.nn import functional as nnf
from transformers import AutoTokenizer, XCLIPModel, XCLIPProcessor


class VideoDescription:
    def __init__(self, config: tp.Dict) -> None:
        self.config = config
        self.init_models()

    def init_models(self) -> None:
        self.encoder = XCLIPModel.from_pretrained(
            self.config["encoder_backbone"]
        )
        self.processor = XCLIPProcessor.from_pretrained(
            self.config["encoder_backbone"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["decoder_backbone"]
        )
        self.decoder = GPT2_Decoder(
            prefix_length=int(self.config["prefix_length"]),
            backbone=self.config["decoder_backbone"],
            prefix_size=int(self.config["prefix_size"]),
        ).to(self.config["device"])
        # self.decoder.load_state_dict(
        #     torch.load(
        #         self.config["model_path"], map_location=self.config["device"]
        #     )
        # )

    def _filter_ngrams(self, output_text: str) -> str:
        a_pos = output_text.find(" A:")
        sec_a_pos = output_text.find(" A:", a_pos + 1)
        return output_text[:sec_a_pos]

    def _generate(
        self,
        model: GPT2_Decoder,
        tokenizer: AutoTokenizer,
        embed: torch.Tensor,
        stop_token: str = "<|endoftext|>",
    ) -> str:
        model.eval()
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        tokens = torch.tensor(tokenizer.encode(self.config["prompt"]))
        tokens = tokens.unsqueeze(0).to(self.config["device"])

        with torch.no_grad():
            emb_tokens = model.gpt.transformer.wte(tokens)
            if embed is not None:
                generated = torch.cat((embed, emb_tokens), dim=1)
            else:
                generated = emb_tokens
            for _ in range(int(self.config["max_words"])):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (
                    float(self.config["temperature"])
                    if float(self.config["temperature"]) > 0
                    else 1.0
                )
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > float(
                    self.config["top_p"]
                )
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            output_text = self._filter_ngrams(output_text)
            generated_list.append(output_text)

        return generated_list[0]

    def get_caption(self, video: tp.List[Image]) -> tp.Dict[str, str]:
        inputs = self.processor(videos=list(video), return_tensors="pt").to(
            self.config["device"]
        )
        with torch.no_grad():
            embedding = (
                self.encoder.get_video_features(**inputs).cpu().unsqueeze(0)
            )
            prefix_embed = (
                self.decoder.clip_project(embedding)
                .reshape(1, int(self.config["prefix_length"]), -1)
                .to(self.config["device"])
            )
            generated_text_prefix = self._generate(
                model=self.decoder,
                tokenizer=self.tokenizer,
                embed=prefix_embed,
            )

        generated_text_prefix.replace("\n", " ").replace("<|endoftext|", "")
        ans = generated_text_prefix[len(self.config["prompt"]) :].strip()
        return {"answer": ans}
