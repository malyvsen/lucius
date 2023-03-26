import pickle
from pathlib import Path

import click


@click.command()
@click.argument(
    "summary-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
def main(summary_path: Path, out_path: Path | None):
    with summary_path.open("rb") as summary_file:
        summary = pickle.load(summary_file)

    if out_path is None:
        out_path = Path.cwd() / (f"{summary_path.stem}.md")
    out_path.write_text(summary.markdown)


main()
