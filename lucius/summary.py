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

        @property
        def html(self):
            hours = self.segment.content.start // 3600
            minutes = (self.segment.content.start % 3600) // 60
            seconds = self.segment.content.start % 60
            image_html = "<p>" + "\n".join(image.html for image in self.images) + "</p>"
            return f"""
<details>
    <summary>{self.summary}</summary>
    <p>Start time: <code>{hours:02.0f}h {minutes:02.0f}m {seconds:02.0f}s</code></p>
    <div>
        <p>{self.segment.content.text}</p>
    </div>
</details>
{image_html}
        """.strip()

    fragments: list[Fragment]

    @property
    def html(self):
        return "\n".join(fragment.html for fragment in self.fragments)
