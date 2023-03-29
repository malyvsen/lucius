import pickle
from pathlib import Path

import click

from lucius import Webpage


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

    webpage = Webpage(
        title=in_path.stem.replace("-", " ").replace("_", " ").capitalize(),
        inner_html=to_export.html,
    )
    if out_path is None:
        out_path = Path.cwd() / (f"{in_path.stem}.html")
    out_path.write_text(webpage.html)


main()
