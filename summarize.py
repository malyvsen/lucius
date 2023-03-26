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
    sentences = CompoundSegment.assemble_sentences(segments)
    paragraphs = CompoundSegment.combine_segments(sentences, min_duration=60)
    for paragraph in paragraphs:
        summary = summarizer(paragraph.text, max_length=64)[0]["generated_text"]
        print(summary)


main()
