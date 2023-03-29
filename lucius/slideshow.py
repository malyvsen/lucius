from dataclasses import dataclass
from io import FileIO

from pypdf import PageObject, PdfReader
from tqdm import tqdm

from .images import Image


@dataclass(frozen=True)
class Slideshow:
    @dataclass(frozen=True)
    class Page:
        images: list[Image]

        @classmethod
        def from_pdf_page(cls, pdf_page: PageObject):
            return cls(
                images=[Image(data=pdf_image.data) for pdf_image in pdf_page.images],
            )

    pages: list[Page]

    @classmethod
    def from_pdf(
        cls,
        pdf: FileIO,
    ):
        reader = PdfReader(pdf)
        return cls(
            pages=[
                cls.Page.from_pdf_page(pdf_page)
                for pdf_page in tqdm(reader.pages, desc="Loading slides")
            ]
        )
