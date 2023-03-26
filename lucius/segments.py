from dataclasses import dataclass, field
from typing import Iterable, Sequence

from faster_whisper.transcribe import Segment as WhisperSegment


@dataclass(frozen=True)
class BaseSegment:
    start: float = field(init=False)
    end: float = field(init=False)
    text: str = field(init=False)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def legible(self):
        return f"[{self.start} => {self.end}] {self.text}"


@dataclass(frozen=True)
class TextSegment(BaseSegment):
    start: float = field(init=True)
    end: float = field(init=True)
    text: str = field(init=True)

    @classmethod
    def empty(cls, moment: float):
        return cls(start=moment, end=moment, text="")

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
    def assemble_sentences(cls, segments: Iterable[BaseSegment]):
        constituents: list[BaseSegment] = []
        for segment in segments:
            constituents.append(segment)
            if any(segment.text.endswith(symbol) for symbol in ".?!"):
                yield cls(constituents=constituents)
                constituents = []
        if len(constituents) != 0:
            yield cls(constituents=constituents)

    @classmethod
    def take_last(cls, segments: Sequence[BaseSegment], min_duration: float):
        constituents = []
        for segment in reversed(segments):
            constituents = [segment] + constituents
            if constituents[-1].end - constituents[0].start >= min_duration:
                break
        return cls(constituents)


@dataclass(frozen=True)
class SegmentWithContext(BaseSegment):
    context: BaseSegment
    """A segment coming just before the main one."""
    content: BaseSegment

    @property
    def start(self):
        return self.context.start

    @property
    def end(self):
        return self.content.end

    @property
    def text(self):
        return " ".join([self.context.text, self.content.text])

    @classmethod
    def iterate(
        cls,
        segments: Iterable[BaseSegment],
        min_context_duration: float,
        min_content_duration: float,
    ):
        context: BaseSegment = TextSegment.empty(moment=0.0)
        content_segments: list[BaseSegment] = []
        for segment in segments:
            content_segments.append(segment)
            content = CompoundSegment(content_segments)
            if content.duration >= min_content_duration:
                yield cls(context=context, content=content)
                context = CompoundSegment.take_last(
                    segments=[context] + content_segments,
                    min_duration=min_context_duration,
                )
                content_segments = []
        if len(content_segments) != 0:
            yield cls(context=context, content=CompoundSegment(content_segments))
