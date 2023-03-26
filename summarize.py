import pickle
from pathlib import Path

import click
from transformers import pipeline

from lucius import CompoundSegment, SegmentWithContext


@click.command()
@click.argument(
    "segments-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
def main(segments_path: Path):
    with segments_path.open("rb") as segments_file:
        segments = pickle.load(segments_file)

    summarizer = pipeline("text2text-generation", model="knkarthick/MEETING_SUMMARY")
    sentences = CompoundSegment.assemble_sentences(segments)
    segments_with_context = SegmentWithContext.iterate(
        sentences, min_context_duration=10, min_content_duration=60
    )
    for segment_with_context in segments_with_context:
        summary = summarizer(segment_with_context.text, max_length=64)[0][
            "generated_text"
        ]
        print(summary)


main()
