"""One-to-one component assignment within a single tracked frame."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from bottle_qc.domain.config import InspectionConfig
from bottle_qc.domain.entities import BottleObservation, Box, Detection


def associate_detections(
    detections: Sequence[Detection],
    *,
    run_id: str,
    config: InspectionConfig,
    observability: Mapping[str, tuple[bool, bool]] | None = None,
    uncertain_track_ids: Collection[str] = (),
) -> list[BottleObservation]:
    """Return observations keyed by tracker identity, never by detection order.

    Observability is supplied by the caller; absence of a component cannot establish it.
    Nearly tied owners make that frame ambiguous for every affected bottle.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must not be empty")
    if detections and any(
        (item.frame_index, item.timestamp) != (detections[0].frame_index, detections[0].timestamp)
        for item in detections
    ):
        raise ValueError("detections must belong to one frame and timestamp")
    bottles = [item for item in detections if item.class_name.casefold() == "bottle"]
    if any(item.track_id is None for item in bottles):
        raise ValueError("bottle detections must have tracker identities")
    if len({item.track_id for item in bottles}) != len(bottles):
        raise ValueError("duplicate bottle track_id in one frame")
    flags = observability or {}
    assignments: dict[str, dict[str, int]] = {bottle.track_id: {} for bottle in bottles}
    ambiguous: set[str] = set()
    edges: list[tuple[float, float, int, str, str]] = []
    for index, component in enumerate(detections):
        kind = component.class_name.casefold()
        if kind not in {"cap", "label"} or component.confidence < config.minimum_confidence:
            continue
        candidates = []
        for bottle in bottles:
            if bottle.confidence < config.minimum_confidence:
                continue
            score = _score(component.box, bottle.box, kind, config.minimum_component_overlap)
            if score is not None:
                candidates.append((score, bottle.track_id))
        candidates.sort()
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= config.association_ambiguity_margin:
            ambiguous.update(track_id for score, track_id in candidates if score - candidates[0][0] <= config.association_ambiguity_margin)
            continue
        # Offer the best owner only; a collision cannot transfer a component to a worse owner.
        score, track_id = candidates[0]
        edges.append((score, -component.confidence, index, track_id, kind))
    for _, _, component_index, track_id, kind in sorted(edges):
        if kind not in assignments[track_id]:
            assignments[track_id][kind] = component_index
    result = []
    for bottle in bottles:
        track_id = bottle.track_id
        cap_clear, label_clear = flags.get(track_id, (False, False))
        result.append(BottleObservation(
            run_id=run_id,
            track_id=track_id,
            bottle=bottle,
            cap_present="cap" in assignments[track_id],
            label_present="label" in assignments[track_id],
            cap_observable=cap_clear,
            label_observable=label_clear,
            ambiguous=track_id in ambiguous,
            identity_uncertain=track_id in uncertain_track_ids,
            reasons=("AMBIGUOUS_COMPONENT_ASSOCIATION",) if track_id in ambiguous else (),
        ))
    return result


def _score(component: Box, bottle: Box, kind: str, minimum_overlap: float) -> float | None:
    width, height = bottle.x2 - bottle.x1, bottle.y2 - bottle.y1
    top, bottom = ((-0.08, 0.28) if kind == "cap" else (0.2, 0.86))
    zone = Box(bottle.x1 - width * 0.08, bottle.y1 + height * top, bottle.x2 + width * 0.08, bottle.y1 + height * bottom)
    cx, cy = component.center
    if not zone.x1 <= cx <= zone.x2 or not zone.y1 <= cy <= zone.y2:
        return None
    overlap = max(0.0, min(component.x2, zone.x2) - max(component.x1, zone.x1)) * max(0.0, min(component.y2, zone.y2) - max(component.y1, zone.y1))
    if overlap / component.area < minimum_overlap:
        return None
    return abs(cx - bottle.center[0]) / width + abs(cy - zone.center[1]) / height + 1 - overlap / component.area
