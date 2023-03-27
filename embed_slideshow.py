import pickle
from pathlib import Path

import click
import easyocr
from sentence_transformers import SentenceTransformer

from lucius import Slideshow


@click.command()
@click.argument(
    "slides-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
@click.option("--use-gpu", type=bool, default=False)
@click.option(
    "--embedding-model-name", default="sentence-transformers/all-mpnet-base-v2"
)
@click.option("--language-code", default="en")
def main(
    slides_path: Path,
    out_path: Path | None,
    use_gpu: bool,
    embedding_model_name: str,
    language_code: str,
):
    ocr_reader = easyocr.Reader([language_code], gpu=use_gpu)
    embedding_model = SentenceTransformer(embedding_model_name)

    with slides_path.open("rb") as pdf_file:
        slideshow = Slideshow.from_pdf(
            pdf_file, ocr_reader=ocr_reader, embedding_model=embedding_model
        )

    if out_path is None:
        out_path = Path.cwd() / f"{slides_path.stem}-slideshow.pkl"
    with out_path.open("wb") as out_file:
        pickle.dump(slideshow, out_file)


main()
