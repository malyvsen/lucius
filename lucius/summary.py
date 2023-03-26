from dataclasses import dataclass

from transformers import Text2TextGenerationPipeline

from .segments import SegmentWithContext


@dataclass(frozen=True)
class Summary:
    @dataclass(frozen=True)
    class Fragment:
        summary: str
        segment: SegmentWithContext

        @property
        def markdown(self):
            hours = self.segment.content.start // 3600
            minutes = (self.segment.content.start % 3600) // 60
            seconds = self.segment.content.start % 60
            return "\n".join(
                [
                    f"- {self.summary}",
                    f"    ",
                    f"    Start time: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}",
                    f"    {self.segment.content.text}",
                ]
            )

        @classmethod
        def generate(
            cls,
            summarizer: Text2TextGenerationPipeline,
            segment: SegmentWithContext,
            max_length: int = 64,
        ):
            return cls(
                summary=summarizer(segment.text, max_length=max_length)[0][
                    "generated_text"
                ],
                segment=segment,
            )

    fragments: list[Fragment]

    @property
    def markdown(self):
        return "\n\n".join(fragment.markdown for fragment in self.fragments)