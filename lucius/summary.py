from dataclasses import dataclass

from numpy.typing import NDArray
from transformers import Text2TextGenerationPipeline

from .images import Image
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
                    f"    `Start time: {hours:02.0f}h {minutes:02.0f}m {seconds:02.0f}s`",
                    f"    ",
                    f"    > {self.segment.content.text}",
                ]
            )

        @classmethod
        def generate(
            cls,
            summarizer: Text2TextGenerationPipeline,
            segment: SegmentWithContext,
            max_tokens: int,
        ):
            return cls(
                summary=summarizer(segment.text, max_length=max_tokens)[0][
                    "generated_text"
                ],
                segment=segment,
            )

    @dataclass(frozen=True)
    class EmbeddedFragment(Fragment):
        embedding: NDArray

    @dataclass(frozen=True)
    class IllustratedFragment(EmbeddedFragment):
        images: list[Image]

    fragments: list[Fragment]

    @property
    def markdown(self):
        return "\n\n".join(fragment.markdown for fragment in self.fragments)
