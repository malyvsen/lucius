import pickle
from pathlib import Path

import click


@click.command()
@click.argument(
    "segments-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
def main(segments_path: Path):
    with segments_path.open("rb") as segments_file:
        segments = pickle.load(segments_file)
    print("\n".join(segment.text.strip() for segment in segments))


main()
