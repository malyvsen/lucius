import pickle
from pathlib import Path

import click
from tqdm import tqdm
from transformers import pipeline

from lucius import SegmentWithContext, Summary, TextSegment, Transcript


@click.command()
@click.argument(
    "transcript-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
@click.option("--model-name", default="philschmid/bart-large-cnn-samsum")
@click.option("--min-context", type=float, default=0)
@click.option("--min-content", type=float, default=800)
@click.option("--max-summary-tokens", type=int, default=200)
def main(
    transcript_path: Path,
    out_path: Path | None,
    model_name: str,
    min_context: float,
    min_content: float,
    max_summary_tokens: int,
):
    with transcript_path.open("rb") as transcript_file:
        transcript: Transcript = pickle.load(transcript_file)

    summarizer = pipeline("text2text-generation", model=model_name)
    sentences = TextSegment.iterate_sentences(transcript.segments)
    segments_with_context = SegmentWithContext.iterate(
        sentences, min_context_chars=min_context, min_content_chars=min_content
    )
    summary = Summary(
        [
            Summary.Fragment.generate(
                summarizer=summarizer, segment=segment, max_tokens=max_summary_tokens
            )
            for segment in tqdm(list(segments_with_context), desc="Summarizing")
        ]
    )
    if out_path is None:
        out_path = Path.cwd() / (
            transcript_path.stem.replace("-transcript", "") + "-summary.pkl"
        )
    with out_path.open("wb") as out_file:
        pickle.dump(summary, out_file)


main()
