"""Laya (System-1 decision model) router: NO fallback to static.

Per research requirement: Laya stands alone. Low confidence or Laya
unavailable yields an explicit `needs_clarification` abstention — never a
silent static substitution — so latency/accuracy numbers stay attributable.

Steps instrumented separately for research comparison:
  1. pre_gate      — free modality check (image/sensor_vector short-circuit;
                     NOT a static decision, just input shaping)
  2. laya_load     — checkpoint resident check / lazy load (amortized)
  3. laya_forward  — single non-autoregressive forward pass
  4. arbitration   — threshold + abstain (no fallback)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .base import BaseRouter, RouterDecision
from .latency import StepClock

# Zero-shot Laya confidences on custom criteria land ~0.1-0.5 and jitter
# run-to-run on identical input (e.g. anomaly text scored 0.13 / 0.22 /
# 0.28 across runs — nondeterministic inference-side, plus uncalibrated
# upstream temperatures, see load warning); fine-tuning raises them toward
# ~0.7+. Default accepts the usable zero-shot band so valid domain queries
# route; raise via env after fine-tuning. Env override: FINAL_LAYA_MIN_CONF.
# Raw confidences are always logged (logs/chat_runs.jsonl, /route/compare),
# so research numbers stay honest whatever the gate is.
LAYA_MIN_CONFIDENCE = float(os.getenv("FINAL_LAYA_MIN_CONF", "0.10"))

_CHOICE_QUESTION = {
    "task": {
        "type": "choice",
        "instructions": "Route this request to the correct specialist adapter.",
        "criteria": {
            "deepfake": "face image, real/fake verdict, spectral or forgery artifacts",
            "anomaly": "TEP sensors XMEAS/XMV, pressure, valve, reactor, fault diagnosis",
            "portfolio": "holdings, returns, rebalance, risk, allocation",
            "general": "general question fitting no specialist",
        },
    }
}


class LayaRouter(BaseRouter):
    """Standalone Laya router. `laya` package is imported lazily so the
    static path never requires it."""

    name = "laya"

    def __init__(self, device: Optional[str] = None,
                 min_confidence: float = LAYA_MIN_CONFIDENCE) -> None:
        self._router = None
        self._device = device
        self._load_error: Optional[str] = None
        self.min_confidence = min_confidence

    def route(self, payload: Dict[str, Any]) -> RouterDecision:
        clock = StepClock()

        with clock.step("pre_gate"):
            shaped = self._shape(payload)

        with clock.step("laya_load"):
            ok = self._ensure_loaded()

        if not ok:
            total = clock.total_ms()
            return RouterDecision(
                router="laya",
                task="needs_clarification",
                confidence=0.0,
                reason=f"laya-unavailable:{self._load_error}",
                latency_ms=total,
                steps=clock.steps_dict(),
                abstained=True,
            )

        with clock.step("laya_forward"):
            raw = self._router.predict(shaped, _CHOICE_QUESTION)

        with clock.step("arbitration"):
            task, confidence, reason, abstained = self._arbitrate(raw)

        total = clock.total_ms()
        return RouterDecision(
            router="laya",
            task=task,
            confidence=confidence,
            reason=reason,
            latency_ms=total,
            steps=clock.steps_dict(),
            abstained=abstained,
            extra={"laya_model": (raw.get("routing") or {}).get("model", "unknown")},
        )

    # -- internals -----------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._router is not None:
            return True
        try:
            from . import _torch_shim  # noqa: F401  (eager fallback, Windows-CPU)
            from laya import Router  # lazy: static path never imports this
            kwargs: Dict[str, Any] = {"preload": True}
            if self._device:
                kwargs["device"] = self._device
            self._router = Router(**kwargs)
            return True
        except Exception as exc:  # laya missing / no weights / no net
            self._load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _shape(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Free input shaping only — modality short-circuits obvious cases
        without spending a forward pass; text goes to Laya untouched."""
        if any(payload.get(k) for k in
               ("image_bytes", "image_base64", "image_path", "pixel_values")):
            return {"modality": "image", "body": "face image for real/fake verdict"}
        values = payload.get("values")
        if isinstance(values, list) and len(values) == 52:
            return {"modality": "sensor_vector",
                    "body": payload.get("text", "TEP 52-channel sensor window")}
        return {"modality": "text",
                "body": payload.get("text") or payload.get("prompt") or ""}

    def _arbitrate(self, raw: Dict[str, Any]) -> tuple:
        try:
            ans = (raw.get("answers") or {}).get("task", {}) or {}
            choice = ans.get("choice", "general")
            conf = float(ans.get("confidence", 0.0))
        except (ValueError, TypeError, AttributeError):
            return ("needs_clarification", 0.0, "laya-malformed-answer", True)
        if conf < self.min_confidence:
            # NO static fallback: explicit abstention keeps numbers attributable.
            return ("needs_clarification", conf,
                    f"laya-lowconf:{choice}@{conf:.2f}<{self.min_confidence}", True)
        return (choice, conf, f"laya:{choice}", False)
