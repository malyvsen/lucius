import pickle
from pathlib import Path

import click


@click.command()
@click.argument(
    "in-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
def main(in_path: Path, out_path: Path | None):
    with in_path.open("rb") as in_file:
        to_export = pickle.load(in_file)

    if out_path is None:
        out_path = Path.cwd() / (f"{in_path.stem}.md")
    out_path.write_text(to_export.markdown)


main()
