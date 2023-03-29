import itertools
import pickle
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from lucius import EmbeddedImage, Embedder, Slideshow, Summary


@click.command()
@click.option(
    "--pdf-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--summary-path",
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
    pdf_path: Path,
    summary_path: Path,
    out_path: Path | None,
    model_name: str,
    pretraining: str,
):
    with pdf_path.open("rb") as pdf_file:
        slideshow = Slideshow.from_pdf(pdf_file)

    with summary_path.open("rb") as summary_file:
        summary: Summary = pickle.load(summary_file)

    embedder = Embedder.from_pretrained(model_name=model_name, pretraining=pretraining)
    embedded_slides = [
        Slideshow.EmbeddedSlide.from_slide(slide, embedder=embedder)
        for slide in tqdm(slideshow.slides, desc="Embedding slideshow")
    ]
    embedded_fragments = [
        Summary.EmbeddedFragment(
            summary=fragment.summary,
            segment=fragment.segment,
            embedding=embedder.embed_text(fragment.summary),
        )
        for fragment in tqdm(summary.fragments, desc="Embedding summary")
    ]

    matched_fragments = [[] for slide in embedded_slides]
    slide_idx = 0
    for fragment_idx, fragment in enumerate(embedded_fragments):
        target_proportion = fragment_idx / len(embedded_fragments)
        if slide_idx / len(embedded_slides) < target_proportion - 0.1:
            slide_idx = int(target_proportion * len(embedded_slides))
        slide_idx = max(
            range(slide_idx, min(slide_idx + 2, len(embedded_slides))),
            key=lambda candidate_idx: np.dot(
                embedded_slides[candidate_idx].render.embedding, fragment.embedding
            ),
        )
        matched_fragments[slide_idx].append(fragment)

    summarized_slideshow = Slideshow(
        slides=[
            Slideshow.SummarizedSlide(
                render=slide.render,
                images=slide.images,
                summary_fragments=distribute_images(
                    fragments=fragments, images=slide.images
                ),
            )
            for slide, fragments in zip(embedded_slides, matched_fragments)
        ]
    )

    if out_path is None:
        out_path = Path.cwd() / (
            summary_path.stem.replace("-summary", "") + "-aligned.pkl"
        )
    with out_path.open("wb") as out_file:
        pickle.dump(summarized_slideshow, out_file)


def distribute_images(
    fragments: list[Summary.EmbeddedFragment], images: list[EmbeddedImage]
):
    if len(fragments) == 0:
        return []

    def best_match(image: EmbeddedImage):
        return max(
            fragments,
            key=lambda fragment: np.dot(image.embedding, fragment.embedding),
        )

    return [
        Summary.IllustratedFragment(
            summary=fragment.summary,
            segment=fragment.segment,
            embedding=fragment.embedding,
            images=[image for image in images if best_match(image) is fragment],
        )
        for fragment in fragments
    ]


main()
