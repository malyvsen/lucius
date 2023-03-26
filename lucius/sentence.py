from dataclasses import dataclass
from typing import Iterable

from faster_whisper.transcribe import Segment


@dataclass(frozen=True)
class Sentence:
    start: float
    end: float
    text: str

    @classmethod
    def iterate_segments(cls, segments: Iterable[Segment]):
        constituents: list[Segment] = []
        for segment in segments:
            constituents.append(segment)
            if any(segment.text.endswith(symbol) for symbol in ".?!"):
                yield cls.from_segments(constituents)
                constituents = []
        if len(constituents) != 0:
            yield cls.from_segments(constituents)

    @classmethod
    def from_segments(cls, segments: Iterable[Segment]):
        return cls(
            start=segments[0].start,
            end=segments[-1].end,
            text=" ".join(segment.text.strip() for segment in segments),
        )
