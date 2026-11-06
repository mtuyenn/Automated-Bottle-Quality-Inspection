"""Per-run, streaming quality decisions without a detector or image dependency."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
import json
import math
import uuid

from bottle_qc.domain.config import InspectionConfig, frame_aspect_ratio_matches
from bottle_qc.domain.entities import (
    BottleObservation, EndReason, InspectionEvent, ObservationReason, Outcome, RunMode,
    validate_frame_index, validate_timestamp,
)


@dataclass(slots=True)
class _Track:
    first_seen: float
    last_seen: float
    entry_armed: bool = False
    entered: bool = False
    exit_armed: bool = False
    previous_in_band: bool = False
    identity_uncertain: bool = False
    valid_count: int = 0
    cap_count: int = 0
    label_count: int = 0
    first_valid: float | None = None
    last_valid: float | None = None
    rejected_reasons: set[str] = field(default_factory=set)
    outcome: Outcome = Outcome.PENDING

    @property
    def duration(self) -> float:
        return 0.0 if self.first_valid is None else self.last_valid - self.first_valid


class InspectionEngine:
    """Create one instance per run. Frame timestamps are source seconds.

    Retains constant-size statistics per track and finalized identities up to maximum_tracks.
    A tracker ID is never reused for another product within a run. New runs use new engines.
    """

    def __init__(
        self, *, run_id: str, config: InspectionConfig, mode: RunMode,
        config_id: str = "default", config_hash: str | None = None,
        model_id: str | None = None, model_hash: str | None = None,
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must not be empty")
        if not isinstance(mode, RunMode):
            raise ValueError("mode must be a RunMode")
        self.run_id, self.config, self.mode = run_id, config, mode
        self.config_id, self.config_hash = config_id, config_hash or config.fingerprint
        self.model_id, self.model_hash = model_id, model_hash
        self._tracks: dict[str, _Track] = {}
        self._frame_index = -1
        self._timestamp = 0.0
        self._closed = False
        self._counters = {Outcome.PASS: 0, Outcome.FAIL: 0, Outcome.UNINSPECTED: 0}

    @property
    def statuses(self) -> dict[str, Outcome]:
        return {track_id: track.outcome for track_id, track in self._tracks.items()}

    @property
    def counters(self) -> dict[str, int]:
        return {outcome.value: count for outcome, count in self._counters.items()}

    def mark_identity_uncertain(self, track_ids: Collection[str]) -> None:
        """Stick provider uncertainty onto known pending tracks, including absent IDs."""
        if self._closed:
            raise RuntimeError("run has already ended")
        if isinstance(track_ids, (str, bytes)) or any(
            not isinstance(track_id, str) or not track_id.strip() for track_id in track_ids
        ):
            raise ValueError("uncertain track IDs must be nonempty strings")
        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is not None and track.outcome == Outcome.PENDING:
                track.identity_uncertain = True

    def process_frame(
        self, *, frame_index: int, timestamp: float, frame_size: tuple[int, int],
        observations: Sequence[BottleObservation],
    ) -> tuple[InspectionEvent, ...]:
        if self._closed:
            raise RuntimeError("run has already ended")
        self._validate_frame(frame_index, timestamp, frame_size, observations)
        # Validate the whole frame before consuming it or expiring any track.
        width, height = frame_size
        new_ids = {item.track_id for item in observations if self._in_roi(item, width, height)}.difference(self._tracks)
        if len(self._tracks) + len(new_ids) > self.config.maximum_tracks:
            raise ValueError("maximum_tracks reached; end the run before accepting more identities")
        self._frame_index, self._timestamp = frame_index, timestamp
        events = []
        for track_id, track in self._tracks.items():
            if track.outcome == Outcome.PENDING and timestamp - track.last_seen > self.config.track_timeout_seconds + 1e-9:
                events.append(self._finalize(track_id, track, "TRACKING_TIMEOUT", forced=True))
        for observation in observations:
            track_id = observation.track_id
            track = self._tracks.get(track_id)
            if track is None:
                if not self._in_roi(observation, width, height):
                    continue
                track = self._tracks[track_id] = _Track(timestamp, timestamp)
            if track.outcome != Outcome.PENDING:
                continue
            track.last_seen = timestamp
            track.identity_uncertain |= (
                observation.identity_uncertain
                or ObservationReason.IDENTITY_UNCERTAIN in observation.reasons
            )
            cx, cy = observation.bottle.box.center
            x, y = cx / width, cy / height
            progress = self.config.progress(x)
            in_band = self.config.roi.y1 <= y <= self.config.roi.y2
            in_roi = in_band and self.config.roi.x1 <= x <= self.config.roi.x2
            if not in_band:
                track.entry_armed = False
                track.exit_armed = False
            if in_roi and progress <= self.config.entry_progress - self.config.line_hysteresis:
                track.entry_armed = True
            if in_roi and track.entry_armed and progress >= self.config.entry_progress + self.config.line_hysteresis:
                track.entered = True
            if in_roi and progress <= self.config.exit_progress - self.config.line_hysteresis:
                track.exit_armed = True
            if track.entered and in_roi and self.config.entry_progress <= progress <= self.config.exit_progress:
                self._observe(track, observation)
            crossed_exit = (
                in_band and track.previous_in_band and track.exit_armed
                and progress >= self.config.exit_progress + self.config.line_hysteresis
            )
            track.previous_in_band = in_band
            if crossed_exit:
                events.append(self._finalize(track_id, track, "EXIT_LINE_CROSSED"))
        return tuple(events)

    def _in_roi(self, observation: BottleObservation, width: int, height: int) -> bool:
        cx, cy = observation.bottle.box.center
        return self.config.roi.x1 <= cx / width <= self.config.roi.x2 and self.config.roi.y1 <= cy / height <= self.config.roi.y2

    def finish(self, *, reason: EndReason, frame_index: int, timestamp: float) -> tuple[InspectionEvent, ...]:
        """Flush unfinished products as UNINSPECTED. Repeated finish calls emit nothing."""
        if self._closed:
            return ()
        if not isinstance(reason, EndReason):
            raise ValueError("reason must be an EndReason")
        validate_frame_index(frame_index)
        validate_timestamp(timestamp)
        if frame_index < self._frame_index or timestamp < self._timestamp:
            raise ValueError("finish cannot precede the last processed frame")
        self._frame_index, self._timestamp, self._closed = frame_index, timestamp, True
        return tuple(
            self._finalize(track_id, track, reason.value, forced=True)
            for track_id, track in self._tracks.items() if track.outcome == Outcome.PENDING
        )

    def _observe(self, track: _Track, observation: BottleObservation) -> None:
        invalid = set(observation.reasons)
        if observation.ambiguous:
            invalid.add("AMBIGUOUS_COMPONENT_ASSOCIATION")
        if observation.identity_uncertain:
            invalid.add("IDENTITY_UNCERTAIN")
        if self.mode == RunMode.REAL and not observation.cap_present:
            invalid.add("CAP_NOT_OBSERVABLE")
        if self.mode == RunMode.REAL and not observation.label_present:
            invalid.add("LABEL_NOT_OBSERVABLE")
        if not observation.cap_observable:
            invalid.add("CAP_NOT_OBSERVABLE")
        if not observation.label_observable:
            invalid.add("LABEL_NOT_OBSERVABLE")
        if observation.bottle.confidence < self.config.minimum_confidence:
            invalid.add("LOW_BOTTLE_CONFIDENCE")
        if invalid:
            track.rejected_reasons.update(invalid)
            return
        track.valid_count += 1
        track.cap_count += observation.cap_present
        track.label_count += observation.label_present
        if track.first_valid is None:
            track.first_valid = observation.timestamp
        track.last_valid = observation.timestamp

    def _finalize(self, track_id: str, track: _Track, reason: str, *, forced: bool = False) -> InspectionEvent:
        reasons = {reason}
        if not track.entered:
            reasons.add("ENTRY_NOT_OBSERVED")
        if track.valid_count < self.config.minimum_observations:
            reasons.add("INSUFFICIENT_OBSERVATIONS")
        if track.duration + 1e-9 < self.config.minimum_observation_duration:
            reasons.add("INSUFFICIENT_DURATION")
        if track.identity_uncertain:
            reasons.add("IDENTITY_UNCERTAIN")
        eligible = (
            not forced and track.entered and not track.identity_uncertain
            and track.valid_count >= self.config.minimum_observations
            and track.duration + 1e-9 >= self.config.minimum_observation_duration
        )
        cap_support = track.cap_count / track.valid_count if track.valid_count else None
        label_support = track.label_count / track.valid_count if track.valid_count else None
        if eligible:
            if cap_support < self.config.cap_support_threshold:
                reasons.add("CAP_MISSING")
            if label_support < self.config.label_support_threshold:
                reasons.add("LABEL_MISSING")
            outcome = Outcome.FAIL if {"CAP_MISSING", "LABEL_MISSING"}.intersection(reasons) else Outcome.PASS
            if outcome == Outcome.PASS:
                reasons.add("COMPONENTS_CONFIRMED")
        else:
            outcome = Outcome.UNINSPECTED
            reasons.update(track.rejected_reasons)
        track.outcome = outcome
        self._counters[outcome] += 1
        identity = json.dumps([self.run_id, track_id], separators=(",", ":"))
        return InspectionEvent(
            event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
            run_id=self.run_id, source_item_id=f"track-{track_id}", track_id=track_id,
            outcome=outcome, mode=self.mode, reason_codes=tuple(sorted(reasons)),
            source_index=self._frame_index, source_timestamp_ms=self._timestamp * 1000,
            first_seen_at=track.first_seen, finalized_at=self._timestamp,
            valid_observation_count=track.valid_count, observation_duration=track.duration,
            cap_support=cap_support, label_support=label_support,
            model_id=self.model_id, model_hash=self.model_hash, config_id=self.config_id, config_hash=self.config_hash,
            extra={"cap_observation_count": track.cap_count, "label_observation_count": track.label_count,
                   "rejected_observation_reasons": sorted(track.rejected_reasons)},
        )

    def _validate_frame(self, frame_index: int, timestamp: float, frame_size: tuple[int, int], observations: Sequence[BottleObservation]) -> None:
        validate_frame_index(frame_index)
        validate_timestamp(timestamp)
        if frame_index <= self._frame_index or timestamp < self._timestamp:
            raise ValueError("frames must have increasing indices and nondecreasing timestamps")
        if len(frame_size) != 2 or any(type(value) is not int or value <= 0 for value in frame_size):
            raise ValueError("frame_size must contain positive integer width and height")
        width, height = frame_size
        if not frame_aspect_ratio_matches(width, height, self.config.frame_aspect_ratio):
            raise ValueError("frame aspect ratio differs from configured ROI; recalibrate explicitly")
        if len({item.track_id for item in observations}) != len(observations):
            raise ValueError("duplicate observation for track_id in one frame")
        for item in observations:
            if item.run_id != self.run_id or item.frame_index != frame_index or item.timestamp != timestamp:
                raise ValueError("observation run/frame/timestamp does not match the current frame")
            box = item.bottle.box
            if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
                raise ValueError("bottle box must be clipped to frame bounds")
