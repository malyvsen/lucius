import pickle
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from lucius import EmbeddedImage, Embedder, Slideshow, Summary


@click.command()
@click.option(
    "--summary-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--pdf-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
@click.option("--model-name", default="ViT-H-14")
@click.option("--pretraining", default="laion2b_s32b_b79k")
def main(
    summary_path: Path,
    pdf_path: Path,
    out_path: Path | None,
    model_name: str,
    pretraining: str,
):
    with summary_path.open("rb") as summary_file:
        summary: Summary = pickle.load(summary_file)

    with pdf_path.open("rb") as pdf_file:
        slideshow = Slideshow.from_pdf(pdf_file)

    embedder = Embedder.from_pretrained(model_name=model_name, pretraining=pretraining)
    embedded_fragments = [
        Summary.EmbeddedFragment(
            summary=fragment.summary,
            segment=fragment.segment,
            embedding=embedder.embed_text(fragment.summary),
        )
        for fragment in tqdm(summary.fragments, desc="Embedding summary fragments")
    ]
    embedded_images = [
        EmbeddedImage(data=image.data, embedding=embedder.embed_image(image.pil_image))
        for image in tqdm(
            [image for page in slideshow.pages for image in page.images],
            desc="Embedding images",
        )
    ]

    filter_embedding = embedder.embed_text("scientific chart")
    filtered_images = [
        image
        for image in embedded_images
        if np.dot(image.embedding, filter_embedding) > 0.14
    ]
    best_matches = {
        image: max(
            embedded_fragments,
            key=lambda fragment: np.dot(image.embedding, fragment.embedding),
        )
        for image in filtered_images
    }

    illustrated_summary = Summary(
        fragments=[
            Summary.IllustratedFragment(
                summary=fragment.summary,
                segment=fragment.segment,
                embedding=fragment.embedding,
                images=[
                    image
                    for image in filtered_images
                    if best_matches[image] is fragment
                ],
            )
            for fragment in embedded_fragments
        ]
    )

    if out_path is None:
        out_path = Path.cwd() / (
            summary_path.stem.replace("-summary", "") + "-illustrated.pkl"
        )
    with out_path.open("wb") as out_file:
        pickle.dump(illustrated_summary, out_file)


main()
