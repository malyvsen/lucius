from dataclasses import dataclass, field
from typing import Iterable

from faster_whisper.transcribe import Segment as WhisperSegment


@dataclass(frozen=True)
class BaseSegment:
    start: float = field(init=False)
    end: float = field(init=False)
    text: str = field(init=False)


@dataclass(frozen=True)
class TextSegment(BaseSegment):
    start: float = field(init=True)
    end: float = field(init=True)
    text: str = field(init=True)

    @classmethod
    def from_whisper_segment(cls, whisper_segment: WhisperSegment):
        return cls(
            start=whisper_segment.start,
            end=whisper_segment.end,
            text=whisper_segment.text.strip(),
        )


@dataclass(frozen=True)
class CompoundSegment(BaseSegment):
    constituents: list[BaseSegment]

    @property
    def start(self):
        return self.constituents[0].start

    @property
    def end(self):
        return self.constituents[-1].end

    @property
    def text(self):
        return " ".join(segment.text for segment in self.constituents)

    @classmethod
    def combine_sentences(cls, segments: Iterable[BaseSegment]):
        constituents: list[BaseSegment] = []
        for segment in segments:
            constituents.append(segment)
            if any(segment.text.endswith(symbol) for symbol in ".?!"):
                yield cls(constituents=constituents)
                constituents = []
        if len(constituents) != 0:
            yield cls(constituents=constituents)
