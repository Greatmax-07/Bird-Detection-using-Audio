"""
Multi-species detection from per-segment scores. Pure numpy, no models, so the
server and the evaluation script share exactly the same logic.

Per 3 s segment and species we have two independent scores:
    head     our MLP head trained on Maharashtra Xeno-canto clips (sigmoid outputs)
    birdnet  BirdNET's own multi-label score for that species
They are blended (w_birdnet), then thresholded with hysteresis:
    * a species is reported only if it reaches t_high somewhere in the file
    * its events then extend over neighbouring segments above t_low
    * segments closer than merge_gap seconds are merged into one event
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DetectParams:
    w_birdnet: float = 0.7
    t_high: float = 0.50
    t_low: float = 0.25
    merge_gap: float = 3.0

    @classmethod
    def from_env(cls) -> "DetectParams":
        d = cls()
        return cls(
            w_birdnet=float(os.environ.get("BIRDEAR_W_BIRDNET", d.w_birdnet)),
            t_high=float(os.environ.get("BIRDEAR_T_HIGH", d.t_high)),
            t_low=float(os.environ.get("BIRDEAR_T_LOW", d.t_low)),
            merge_gap=float(os.environ.get("BIRDEAR_MERGE_GAP", d.merge_gap)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def fuse(p_head: np.ndarray, p_birdnet: np.ndarray, has_birdnet: np.ndarray, w: float) -> np.ndarray:
    """(n, C) scores. Species BirdNET does not know use the head alone."""
    return np.where(has_birdnet[None, :], w * p_birdnet + (1.0 - w) * p_head, p_head)


def detect_events(times: np.ndarray, fused: np.ndarray, params: DetectParams) -> dict[int, list[dict]]:
    """{class_id: [event, ...]}; event = start_time, end_time, confidence (max), segments."""
    out: dict[int, list[dict]] = {}
    for k in range(fused.shape[1]):
        col = fused[:, k]
        if col.size == 0 or col.max() < params.t_high:
            continue
        events: list[dict] = []
        for s in np.flatnonzero(col >= params.t_low):
            start, end, conf = float(times[s, 0]), float(times[s, 1]), float(col[s])
            if events and start - events[-1]["end_time"] <= params.merge_gap:
                e = events[-1]
                e["end_time"] = end
                e["confidence"] = max(e["confidence"], conf)
                e["segments"] += 1
            else:
                events.append({"start_time": start, "end_time": end, "confidence": conf, "segments": 1})
        events = [e for e in events if e["confidence"] >= params.t_high]
        if events:
            out[k] = events
    return out
