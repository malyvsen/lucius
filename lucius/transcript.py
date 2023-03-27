from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel, decode_audio
from tqdm import tqdm

from .segments import TextSegment


@dataclass(frozen=True)
class Transcript:
    language: str
    segments: list[TextSegment]

    @property
    def markdown(self):
        return "\n".join(segment.text for segment in self.segments)

    @classmethod
    def from_audio_file(cls, model: WhisperModel, language: str, audio_path: Path):
        audio = decode_audio(
            audio_path.as_posix(), sampling_rate=model.feature_extractor.sampling_rate
        )
        duration = audio.shape[0] / model.feature_extractor.sampling_rate
        segment_iterable, info = model.transcribe(audio, language=language)

        segments = []
        loading_bar = tqdm(
            desc="Transcribing",
            total=duration,
            bar_format="{l_bar}{bar}| {n:.0f}s/{total:.0f}s [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            unit="second of audio",
        )
        loading_bar.refresh()
        for segment in segment_iterable:
            loading_bar.update(segment.end - loading_bar.n)
            segments.append(TextSegment.from_whisper_segment(segment))
        loading_bar.update(duration - loading_bar.n)
        loading_bar.close()

        return cls(language=language, segments=segments)
