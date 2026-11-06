from __future__ import annotations

from collections.abc import Mapping, Sequence

from PIL import Image, ImageDraw

from bottle_qc.domain.entities import Detection


_COLORS = {"bottle": (42, 91, 139), "cap": (31, 132, 84), "label": (211, 144, 28)}
_OUTCOME_COLORS = {"PASS": (31, 132, 84), "FAIL": (190, 43, 43), "UNINSPECTED": (190, 119, 22)}


def render_overlay(
    image: Image.Image,
    detections: Sequence[Detection],
    *,
    mode: str,
    outcomes: Mapping[str, str] | None = None,
) -> Image.Image:
    """Draw source detections and outcomes on a copy; never changes a decision."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 34), fill=(24, 38, 56))
    draw.text((12, 10), mode, fill=(255, 255, 255))
    for item in detections:
        color = _COLORS.get(item.class_name.casefold(), (80, 80, 80))
        box = tuple(int(value) for value in (item.box.x1, item.box.y1, item.box.x2, item.box.y2))
        draw.rectangle(box, outline=color, width=2)
        label = item.class_name
        if item.track_id:
            label += f" #{item.track_id}"
        draw.text((box[0] + 2, max(35, box[1] - 12)), label, fill=color)
    if outcomes:
        for index, (track_id, outcome) in enumerate(outcomes.items()):
            color = _OUTCOME_COLORS[outcome]
            draw.text((12, 42 + 18 * index), f"Bottle {track_id}: {outcome}", fill=color)
    return output
