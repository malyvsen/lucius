from dataclasses import dataclass
from functools import cached_property
from io import BytesIO

from numpy.typing import NDArray
from PIL import Image as PIL_Image


@dataclass(frozen=True)
class Image:
    data: bytes

    @cached_property
    def pil_image(self):
        return PIL_Image.open(BytesIO(self.data))


@dataclass(frozen=True)
class EmbeddedImage(Image):
    embedding: NDArray
