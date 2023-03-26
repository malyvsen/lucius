import pickle
from pathlib import Path

import click
from faster_whisper import WhisperModel

from lucius import transcribe


@click.command()
@click.option(
    "--in-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
@click.option("--model-name", default="medium.en")
@click.option("--device", default="cpu")
@click.option("--precision", default="int8")
def main(
    in_path: Path, out_path: Path | None, model_name: str, device: str, precision: str
):
    model = WhisperModel(model_name, device=device, compute_type=precision)
    segments = transcribe(model=model, audio_path=in_path)

    if out_path is None:
        out_path = Path.cwd() / f"{in_path.stem}-segments.pkl"
    with out_path.open("wb") as segments_file:
        pickle.dump(list(segments), segments_file)


main()
