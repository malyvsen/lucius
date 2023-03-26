from pathlib import Path

from faster_whisper import WhisperModel, decode_audio
from tqdm import tqdm

from .segments import TextSegment


def transcribe(model: WhisperModel, audio_path: Path):
    audio = decode_audio(
        audio_path.as_posix(), sampling_rate=model.feature_extractor.sampling_rate
    )
    duration = audio.shape[0] / model.feature_extractor.sampling_rate
    segments, info = model.transcribe(audio)

    loading_bar = tqdm(
        desc="Transcribing",
        total=duration,
        bar_format="{l_bar}{bar}| {n:.0f}s/{total:.0f}s [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        unit="s",
    )
    for segment in segments:
        loading_bar.update(segment.end - loading_bar.n)
        yield TextSegment.from_whisper_segment(segment)
    loading_bar.update(duration - loading_bar.n)
    loading_bar.close()
