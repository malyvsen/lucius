from dataclasses import dataclass
from typing import Callable

import open_clip
import torch
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class Embedder:
    model: open_clip.CLIP
    image_transform: Callable[[Image.Image], torch.Tensor]
    tokenizer: Callable[[str], torch.Tensor]

    @classmethod
    def from_pretrained(cls, model_name: str, pretraining: str):
        model, _, image_transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretraining
        )
        return cls(
            model=model,
            image_transform=image_transform,
            tokenizer=open_clip.get_tokenizer(model_name),
        )

    def embed_text(self, text: str):
        tokens = self.tokenizer(text)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
        return self.normalize(embeddings)[0]

    def embed_image(self, image: Image.Image):
        transformed = self.image_transform(image).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.model.encode_image(transformed)
        return self.normalize(embeddings)[0]

    def normalize(self, embeddings: torch.Tensor) -> NDArray:
        normalized = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return normalized.detach().cpu().numpy()
