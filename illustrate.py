import itertools
import pickle
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm, trange

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
@click.option("--num-attempts", type=int, default=100_000)
def main(
    pdf_path: Path,
    summary_path: Path,
    out_path: Path | None,
    model_name: str,
    pretraining: str,
    num_attempts: int,
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

    def random_alignment() -> list[int]:
        steps = np.random.uniform(low=0, high=1, size=len(embedded_fragments))
        unbound_indices = np.cumsum(steps)
        return (
            np.round(unbound_indices / unbound_indices[-1] * (len(embedded_slides) - 1))
            .astype(int)
            .tolist()
        )

    best_alignment = max(
        (
            random_alignment()
            for attempt in trange(num_attempts, desc="Searching alignments")
        ),
        key=lambda alignment: sum(
            np.dot(
                embedded_slides[slide_idx].render.embedding,
                embedded_fragments[fragment_idx].embedding,
            )
            for fragment_idx, slide_idx in enumerate(alignment)
        ),
    )

    aligned_fragments: list[list[Summary.EmbeddedFragment]] = [
        [] for slide in embedded_slides
    ]
    for idx, fragment in zip(best_alignment, embedded_fragments):
        aligned_fragments[idx].append(fragment)
    summarized_slideshow = Slideshow(
        slides=[
            Slideshow.SummarizedSlide(
                render=slide.render,
                images=slide.images,
                summary_fragments=fragments,
            )
            for slide, fragments in zip(embedded_slides, aligned_fragments)
        ]
    )

    if out_path is None:
        out_path = Path.cwd() / (
            summary_path.stem.replace("-summary", "") + "-illustrated.pkl"
        )
    with out_path.open("wb") as out_file:
        pickle.dump(summarized_slideshow, out_file)


main()
