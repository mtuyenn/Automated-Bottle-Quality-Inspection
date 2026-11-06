from __future__ import annotations

from collections.abc import Sequence
import json
import uuid

from bottle_qc.domain.config import InspectionConfig
from bottle_qc.domain.entities import BottleObservation, InspectionEvent, ObservationReason, Outcome, RunMode


def inspect_image(
    observations: Sequence[BottleObservation],
    *,
    run_id: str,
    source_item_id: str,
    config: InspectionConfig,
    mode: RunMode,
    config_id: str,
    config_hash: str,
    model_id: str | None = None,
    model_hash: str | None = None,
    source_index: int = 0,
    source_timestamp_ms: float | None = 0.0,
) -> tuple[InspectionEvent, ...]:
    """Return independent image-local outcomes without temporal track state."""
    if not isinstance(run_id, str) or not run_id.strip() or not isinstance(source_item_id, str) or not source_item_id.strip():
        raise ValueError("run_id and source_item_id must not be empty")
    if not isinstance(mode, RunMode):
        raise ValueError("mode must be a RunMode")
    if type(source_index) is not int or source_index < 0:
        raise ValueError("source_index must be a nonnegative integer")
    if source_timestamp_ms is not None and (type(source_timestamp_ms) not in (int, float) or source_timestamp_ms < 0):
        raise ValueError("source_timestamp_ms must be nonnegative or None")
    events = []
    seen = set()
    for observation in observations:
        if observation.run_id != run_id:
            raise ValueError("image observation belongs to another run")
        if observation.track_id in seen:
            raise ValueError("duplicate bottle identity in one image")
        seen.add(observation.track_id)
        reasons = set(observation.reasons)
        if observation.ambiguous:
            reasons.add(ObservationReason.AMBIGUOUS_COMPONENT_ASSOCIATION.value)
        if observation.identity_uncertain:
            reasons.add(ObservationReason.IDENTITY_UNCERTAIN.value)
        if not observation.cap_observable:
            reasons.add(ObservationReason.CAP_NOT_OBSERVABLE.value)
        if not observation.label_observable:
            reasons.add(ObservationReason.LABEL_NOT_OBSERVABLE.value)
        if observation.bottle.confidence < config.minimum_confidence:
            reasons.add(ObservationReason.LOW_BOTTLE_CONFIDENCE.value)
        eligible = not reasons.intersection({
            ObservationReason.AMBIGUOUS_COMPONENT_ASSOCIATION.value,
            ObservationReason.IDENTITY_UNCERTAIN.value,
            ObservationReason.CAP_NOT_OBSERVABLE.value,
            ObservationReason.LABEL_NOT_OBSERVABLE.value,
            ObservationReason.LOW_BOTTLE_CONFIDENCE.value,
        })
        if eligible:
            if not observation.cap_present:
                reasons.add("CAP_MISSING")
            if not observation.label_present:
                reasons.add("LABEL_MISSING")
            outcome = Outcome.FAIL if {"CAP_MISSING", "LABEL_MISSING"}.intersection(reasons) else Outcome.PASS
            if outcome == Outcome.PASS:
                reasons.add("COMPONENTS_CONFIRMED")
        else:
            outcome = Outcome.UNINSPECTED
        item_id = f"{source_item_id}:bottle:{observation.track_id}"
        event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([run_id, item_id], separators=(",", ":"))))
        events.append(InspectionEvent(
            event_id=event_id,
            run_id=run_id,
            source_item_id=item_id,
            track_id=observation.track_id,
            outcome=outcome,
            mode=mode,
            reason_codes=tuple(sorted(reasons)),
            source_index=source_index,
            source_timestamp_ms=source_timestamp_ms,
            first_seen_at=observation.timestamp,
            finalized_at=observation.timestamp,
            valid_observation_count=1 if eligible else 0,
            observation_duration=0.0,
            cap_support=float(observation.cap_present) if observation.cap_observable else None,
            label_support=float(observation.label_present) if observation.label_observable else None,
            model_id=model_id,
            model_hash=model_hash,
            config_id=config_id,
            config_hash=config_hash,
            extra={"source_kind": "image", "image_local_identity": observation.track_id},
        ))
    return tuple(events)
