from dataclasses import dataclass
from io import FileIO

from numpy.typing import NDArray
from pdf2image import convert_from_bytes
from pypdf import PageObject, PdfReader

from .embedder import Embedder
from .images import EmbeddedImage, Image
from .summary import Summary


@dataclass(frozen=True)
class Slideshow:
    @dataclass(frozen=True)
    class Slide:
        render: Image
        images: list[Image]

        @property
        def html(self):
            return f"<p><aside>{self.render.html}</aside></p>"

        @classmethod
        def from_pdf_page(cls, render: Image, pdf_page: PageObject):
            try:
                pdf_images = pdf_page.images
            except ValueError:
                pdf_images = []
            return cls(
                render=render,
                images=[Image.from_data(pdf_image.data) for pdf_image in pdf_images],
            )

    @dataclass(frozen=True)
    class EmbeddedSlide(Slide):
        render: EmbeddedImage
        images: list[EmbeddedImage]

        @classmethod
        def from_slide(cls, slide: "Slideshow.Slide", embedder: Embedder):
            return cls(
                render=EmbeddedImage.from_image(slide.render, embedder=embedder),
                images=[
                    EmbeddedImage.from_image(image, embedder=embedder)
                    for image in slide.images
                ],
            )

    @dataclass(frozen=True)
    class SummarizedSlide(EmbeddedSlide):
        summary_fragments: list[Summary.Fragment]

        @property
        def html(self):
            fragment_html = "\n".join(
                fragment.html for fragment in self.summary_fragments
            )
            return f"""
                <p><aside>{self.render.html}</aside></p>
                {fragment_html}
            """

    slides: list[Slide]

    @property
    def html(self):
        return "\n".join(slide.html for slide in self.slides)

    @classmethod
    def from_pdf(cls, pdf: FileIO):
        renders = convert_from_bytes(pdf.read())
        reader = PdfReader(pdf)
        return cls(
            slides=[
                cls.Slide.from_pdf_page(
                    render=Image(pil_image=render), pdf_page=pdf_page
                )
                for render, pdf_page in zip(renders, reader.pages)
            ]
        )
