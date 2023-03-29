import base64
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PIL_Image


@dataclass(frozen=True)
class Image:
    data: bytes

    @cached_property
    def pil_image(self):
        return PIL_Image.open(BytesIO(self.data))

    @cached_property
    def html(self):
        buffer = BytesIO()
        self.pil_image.save(buffer, format="PNG")
        encoding = base64.b64encode(buffer.getvalue()).decode()
        return f"<aside><img src=data:image/png;base64,{encoding}></aside>"


@dataclass(frozen=True)
class EmbeddedImage(Image):
    embedding: NDArray

    def __eq__(self, other: "EmbeddedImage"):
        return self.data == other.data and np.all(self.embedding == other.embedding)

    def __hash__(self):
        return hash((self.data, tuple(self.embedding)))
