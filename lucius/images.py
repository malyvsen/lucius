import base64
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PIL_Image

from .embedder import Embedder


@dataclass(frozen=True)
class Image:
    pil_image: PIL_Image.Image

    @cached_property
    def html(self):
        buffer = BytesIO()
        self.pil_image.save(buffer, format="PNG")
        encoding = base64.b64encode(buffer.getvalue()).decode()
        return f"<img src=data:image/png;base64,{encoding}>"

    @classmethod
    def from_data(cls, data: bytes):
        return cls(pil_image=PIL_Image.open(BytesIO(data)))


@dataclass(frozen=True)
class EmbeddedImage(Image):
    embedding: NDArray

    @classmethod
    def from_image(cls, image: Image, embedder: Embedder):
        return cls(
            pil_image=image.pil_image,
            embedding=embedder.embed_image(image.pil_image),
        )
