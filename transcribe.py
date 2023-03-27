import pickle
from pathlib import Path

import click
from faster_whisper import WhisperModel

from lucius import Transcript


@click.command()
@click.argument(
    "audio-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
@click.option("--language", default="en")
@click.option("--model-name", default="large-v2")
@click.option("--device", default="cpu")
@click.option("--precision", default="int8")
def main(
    audio_path: Path,
    out_path: Path | None,
    language: str,
    model_name: str,
    device: str,
    precision: str,
):
    model = WhisperModel(model_name, device=device, compute_type=precision)
    transcript = Transcript.from_audio_file(
        model=model, language=language, audio_path=audio_path
    )

    if out_path is None:
        out_path = Path.cwd() / f"{audio_path.stem}-transcript.pkl"
    with out_path.open("wb") as out_file:
        pickle.dump(transcript, out_file)


main()
