import pickle
from pathlib import Path

import click
from transformers import pipeline

from lucius import CompoundSegment


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
    for sentence in CompoundSegment.combine_sentences(segments):
        summary = summarizer(f"Lecturer: {sentence.text}", max_length=64)[0][
            "generated_text"
        ]
        print(summary)


main()
