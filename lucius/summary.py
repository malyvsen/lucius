from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from transformers import Text2TextGenerationPipeline

from .images import Image
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

        def __eq__(self, other: "Summary.EmbeddedFragment"):
            return (
                self.summary == other.summary
                and self.segment == other.segment
                and np.all(self.embedding == other.embedding)
            )

        def __hash__(self):
            return hash((self.summary, self.segment, tuple(self.embedding)))

    @dataclass(frozen=True)
    class IllustratedFragment(EmbeddedFragment):
        images: list[Image]

        @property
        def html(self):
            image_html = "<p>" + "\n".join(image.html for image in self.images) + "</p>"
            return f"""
                <details>
                    <summary>{self.summary}</summary>
                    {self.segment.html}
                    {image_html}
                </details>
            """

    fragments: list[Fragment]

    @property
    def html(self):
        return "\n".join(fragment.html for fragment in self.fragments)
