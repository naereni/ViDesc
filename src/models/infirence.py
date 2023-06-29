import torch
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer, XCLIPModel, XCLIPProcessor

from src.models.config import Config
from src.models.model import GPT2_Decoder
from src.models.utils import read_video

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class Pipeline:
    def __init__(self) -> None:
        self.config = Config()
        self.init_models()

    def init_models(self) -> None:
        self.encoder = XCLIPModel.from_pretrained(self.config.encoder_backbone)
        self.processor = XCLIPProcessor.from_pretrained(
            self.config.encoder_backbone
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.config.decoder_backbone
        )
        self.decoder = GPT2_Decoder(
            prefix_length=self.config.prefix_length,
            backbone=self.config.decoder_backbone,
            prefix_size=self.config.prefix_size,
        )
        self.decoder.load_state_dict(
            torch.load(self.config.model_path, map_location="cpu")
        )

    def _filter_ngrams(self, output_text: str) -> str:
        a_pos = output_text.find(" A:")
        sec_a_pos = output_text.find(" A:", a_pos + 1)
        return output_text[:sec_a_pos]

    def _generate(
        self,
        model: GPT2_Decoder,
        tokenizer: GPT2Tokenizer,
        embed: torch.Tensor,
        stop_token: str = "<|endoftext|>",
    ) -> str:
        model.eval()
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        # DEVICE = next(model.parameters()).DEVICE
        tokens = torch.tensor(tokenizer.encode(self.config.prompt))
        tokens = tokens.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb_tokens = model.gpt.transformer.wte(tokens)
            if embed is not None:
                generated = torch.cat((embed, emb_tokens), dim=1)
            else:
                generated = emb_tokens
            for _ in range(self.config.max_words):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (
                    self.config.temperature
                    if self.config.temperature > 0
                    else 1.0
                )
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
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

    def get_caption(self, input_path: str) -> dict[str, str]:
        video = read_video(self.config, input_path)
        inputs = self.processor(videos=list(video), return_tensors="pt").to(
            DEVICE
        )
        with torch.no_grad():
            embedding = (
                self.encoder.get_video_features(**inputs).cpu().unsqueeze(0)
            )
            prefix_embed = (
                self.decoder.clip_project(embedding)
                .reshape(1, self.config.prefix_length, -1)
                .to(DEVICE)
            )
            generated_text_prefix = self._generate(
                model=self.decoder,
                tokenizer=self.tokenizer,
                embed=prefix_embed,
            )

        generated_text_prefix.replace("\n", " ").replace("<|endoftext|", "")
        ans = generated_text_prefix[len(self.config.prompt) :].strip()
        return {"answer": ans}


def test_capt():
    return "test success"


# pp = Pipeline()
# ans = pp.get_caption("/home/naer/vids/test_video.mp4")
