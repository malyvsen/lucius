from dataclasses import dataclass

from numpy.typing import NDArray
from transformers import Text2TextGenerationPipeline

from .segments import BaseSegment


@dataclass(frozen=True)
class Summary:
    @dataclass(frozen=True)
    class Fragment:
        summary: str
        segment: BaseSegment

        @classmethod
        def generate(
            cls,
            summarizer: Text2TextGenerationPipeline,
            segment: BaseSegment,
            max_tokens: int,
        ):
            return cls(
                summary=summarizer(segment.text, max_length=max_tokens)[0][
                    "generated_text"
                ],
                segment=segment,
            )

        @property
        def html(self):
            return f"""
                <details>
                    <summary>{self.summary}</summary>
                    {self.segment.html}
                </details>
            """

    @dataclass(frozen=True)
    class EmbeddedFragment(Fragment):
        embedding: NDArray

    fragments: list[Fragment]

    @property
    def html(self):
        return "\n".join(fragment.html for fragment in self.fragments)
