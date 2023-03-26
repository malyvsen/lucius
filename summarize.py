import pickle
from pathlib import Path

import click
from tqdm import tqdm
from transformers import pipeline

from lucius import CompoundSegment, SegmentWithContext, Summary


@click.command()
@click.argument(
    "segments-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(file_okay=False, dir_okay=False, writable=True, path_type=Path),
    default=None,
)
@click.option("--model-name", default="knkarthick/MEETING_SUMMARY")
@click.option("--min-context", type=float, default=10.0)
@click.option("--min-content", type=float, default=60.0)
def main(
    segments_path: Path,
    out_path: Path | None,
    model_name: str,
    min_context: float,
    min_content: float,
):
    with segments_path.open("rb") as segments_file:
        segments = pickle.load(segments_file)

    summarizer = pipeline("text2text-generation", model=model_name)
    sentences = CompoundSegment.assemble_sentences(segments)
    segments_with_context = SegmentWithContext.iterate(
        sentences, min_context_duration=min_context, min_content_duration=min_content
    )
    summary = Summary(
        [
            Summary.Fragment.generate(summarizer=summarizer, segment=segment)
            for segment in tqdm(list(segments_with_context), desc="Summarizing")
        ]
    )
    if out_path is None:
        out_path = Path.cwd() / (
            segments_path.stem.replace("-segments", "") + "-summary.pkl"
        )
    with out_path.open("wb") as out_file:
        pickle.dump(summary, out_file)


main()
