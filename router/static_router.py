"""Static (rule-based) router: deterministic, zero model load, fully offline.

Steps instrumented separately for research comparison:
  1. modality_gate  — image bytes / 52-float vector / holdings JSON / text
  2. keyword_vote   — per-task lexicon hits over text fields
  3. evidence_probe — structured keys (detector_evidence, XMEAS_/XMV_, telemetry)
  4. arbitration    — combine signals into task + confidence
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import BaseRouter, RouterDecision
from .latency import StepClock

_IMAGE_KEYS = ("image_bytes", "image_base64", "image_path", "pixel_values")
_SENSOR_RE = re.compile(r"\b(XMEAS_\d{1,2}|XMV_\d{1,2})\b", re.IGNORECASE)

LEXICONS: Dict[str, List[str]] = {
    "deepfake": ["deepfake", "face", "fake", "real", "spectral", "fft", "forgery",
                 "synthetic", "manipulated", "face-swap", "faceswap"],
    "anomaly": ["tep", "sensor", "anomaly", "fault", "pressure", "valve", "reactor",
                "condenser", "separator", "stripper", "cooling", "xmeas", "xmv",
                "telemetry", "threshold", "cva", "lstm"],
    "portfolio": ["portfolio", "holdings", "returns", "rebalance", "risk", "sharpe",
                  "allocation", "equity", "drawdown", "position"],
}

ABSTAIN_THRESHOLD = 0.35


def _text_fields(payload: Dict[str, Any]) -> str:
    parts = []
    for key in ("text", "prompt", "question", "caption"):
        val = payload.get(key)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(parts).lower()


class StaticRouter(BaseRouter):
    name = "static"

    def route(self, payload: Dict[str, Any]) -> RouterDecision:
        clock = StepClock()

        with clock.step("modality_gate"):
            modality = self._modality(payload)

        with clock.step("keyword_vote"):
            votes = self._vote(_text_fields(payload))

        with clock.step("evidence_probe"):
            evidence_hits = self._evidence(payload)

        with clock.step("arbitration"):
            task, confidence, reason = self._arbitrate(modality, votes, evidence_hits)

        total = clock.total_ms()
        steps = clock.steps_dict()
        return RouterDecision(
            router="static",
            task=task,
            confidence=confidence,
            reason=reason,
            latency_ms=total,
            steps=steps,
            abstained=(task == "needs_clarification"),
        )

    # -- steps ---------------------------------------------------------
    def _modality(self, payload: Dict[str, Any]) -> str:
        if any(payload.get(k) for k in _IMAGE_KEYS):
            return "image"
        values = payload.get("values")
        if isinstance(values, list) and len(values) == 52 and all(
            isinstance(v, (int, float)) for v in values
        ):
            return "sensor_vector"
        text = _text_fields(payload)
        if any(k in text for k in ("holdings", "portfolio", "rebalance", "sharpe")):
            return "portfolio_json"
        if "values" in payload or "sensors" in payload or "telemetry" in payload:
            return "telemetry_dict"
        return "text"

    def _vote(self, text: str) -> Dict[str, int]:
        return {
            task: sum(1 for kw in kws if kw in text)
            for task, kws in LEXICONS.items()
        }

    def _evidence(self, payload: Dict[str, Any]) -> Dict[str, bool]:
        text = _text_fields(payload)
        keys = set(payload.keys())
        return {
            "sensor_ids": bool(_SENSOR_RE.search(text)),
            "detector_evidence": "detector_evidence" in keys,
            "telemetry": any(k in keys for k in ("telemetry", "sensors", "values")),
            "portfolio_keys": any(
                k in keys for k in ("holdings", "positions", "returns")
            ),
        }

    def _arbitrate(self, modality: str, votes: Dict[str, int],
                   evidence: Dict[str, bool]) -> tuple:
        if modality == "image":
            return ("deepfake", 0.95, "modality=image")
        if modality == "sensor_vector" or (
            modality == "telemetry_dict"
            and (evidence["sensor_ids"] or evidence["detector_evidence"])
        ):
            return ("anomaly", 0.93, "modality=sensor-telemetry+evidence")
        if modality == "portfolio_json" or evidence["portfolio_keys"]:
            return ("portfolio", 0.9, "modality=portfolio-holdings")
        best = max(votes, key=votes.get)
        hits = votes[best]
        if hits >= 2:
            return (best, min(0.85, 0.55 + 0.1 * hits), f"keyword={best}:{hits}-hits")
        if hits == 1:
            return (best, 0.45, f"keyword={best}:1-hit")
        return ("needs_clarification", 0.2, "no-modality-no-keywords")
