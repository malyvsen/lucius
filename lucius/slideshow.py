from dataclasses import dataclass
from functools import cached_property
from io import BytesIO, FileIO

import easyocr
from numpy.typing import NDArray
from PIL import Image as PIL_Image
from pypdf import PageObject, PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass(frozen=True)
class Slideshow:
    pages: list["Page"]

    @classmethod
    def from_pdf(
        cls,
        pdf: FileIO,
        ocr_reader: easyocr.Reader,
        embedding_model: SentenceTransformer,
    ):
        reader = PdfReader(pdf)
        return cls(
            pages=[
                Page.from_pdf_page(
                    pdf_page, ocr_reader=ocr_reader, embedding_model=embedding_model
                )
                for pdf_page in tqdm(reader.pages, desc="Embedding slides")
            ]
        )


@dataclass(frozen=True)
class Page:
    text: "EmbeddedText"
    images: list["Image"]

    @classmethod
    def from_pdf_page(
        cls,
        pdf_page: PageObject,
        ocr_reader: easyocr.Reader,
        embedding_model: SentenceTransformer,
    ):
        return cls(
            text=EmbeddedText.embed(
                pdf_page.extract_text(), embedding_model=embedding_model
            ),
            images=[
                Image.from_data(
                    data=pdf_image.data,
                    ocr_reader=ocr_reader,
                    embedding_model=embedding_model,
                )
                for pdf_image in pdf_page.images
            ],
        )


@dataclass(frozen=True)
class Image:
    data: bytes
    text: list["EmbeddedText"]

    @cached_property
    def pil_image(self):
        return PIL_Image.open(BytesIO(self.data))

    @classmethod
    def from_data(
        cls,
        data: bytes,
        ocr_reader: easyocr.Reader,
        embedding_model: SentenceTransformer,
    ):
        ocr_results = ocr_reader.readtext(data, detail=0)
        return cls(
            data=data,
            text=EmbeddedText.embed_many(ocr_results, embedding_model=embedding_model),
        )


@dataclass(frozen=True)
class EmbeddedText:
    text: str
    embedding: NDArray

    @classmethod
    def embed(cls, text: str, embedding_model: SentenceTransformer):
        return cls(text=text, embedding=embedding_model.encode(text))

    @classmethod
    def embed_many(cls, texts: list[str], embedding_model: SentenceTransformer):
        embeddings = embedding_model.encode(texts)
        return [
            cls(text=text, embedding=embedding)
            for text, embedding in zip(texts, embeddings, strict=True)
        ]
