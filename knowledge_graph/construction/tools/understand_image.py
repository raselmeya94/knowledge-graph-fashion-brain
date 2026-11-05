"""Image understanding through the Ollama vision model."""

from __future__ import annotations

from knowledge_graph.construction.detect import payload_bytes
from knowledge_graph.errors import ModelUnavailableError, ValidationFailed

_PROMPT = (
    "Describe this image as plain text for a knowledge graph. "
    "Name the people, organizations, products, materials, places, and actions you can actually see. "
    "Do not invent text that is not visible."
)


class UnderstandImageTool:
    name = "understand_image"
    description = "Ask Ollama to describe an image. The description is the text passed to relationship identification."

    def run(self, payload: dict, ctx) -> dict:
        data = payload_bytes(payload)
        if not data:
            raise ValidationFailed("Image bytes are required.")
        filename = str(payload.get("filename") or "image")
        try:
            text = ctx.ollama.describe_image(data, prompt=_PROMPT).strip()
        except ModelUnavailableError as exc:
            return {
                "kind": "image",
                "title": filename,
                "text": "",
                "warnings": [exc.message],
            }
        return {"kind": "image", "title": filename, "text": text, "warnings": []}
