"""Fine-tuned LAYA router for live chat: BASE vs DEEPFAKE, decided per turn.

This is the checkpoint trained in `C:/Users/Admin/Desktop/Router`
(`models/laya_router`, SHA-256 cbca78ad…): a two-route Laya head fitted on the
five-field runtime state and scored against its own `questions.json`. It is
consulted for every non-`deep_thinking` turn — text or image — and its choices
are kept exactly as trained:

  BASE      answer with the shared base model (no specialist adapter)
  DEEPFAKE  run the deepfake forensic pipeline on the active image

There are no keyword cues, no modality short-circuits and no preset replies
here: if the checkpoint cannot answer, the turn goes to BASE with an explicit
`gate-unavailable` reason rather than an invented specialist choice.

DEEPFAKE is only *executable* when the active asset is available and supported
(Router's `supported_asset_exists` contract). The four-route zero-shot router
stays in `LayaRouter`, where the anomaly endpoint still uses it.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base import BaseRouter, RouterDecision
from .laya_gate import LayaGate, build_state, serialize, supported_asset_exists

# How many of the newest turns a routing view may consider. The checkpoint was
# trained with short history; the newest request is always kept in full.
_MAX_HISTORY_TURNS = 6
_MAX_TURN_CHARS = 700


class TrainedLayaRouter(BaseRouter):
    """The application's primary router: the fine-tuned BASE/DEEPFAKE head."""

    name = "laya-finetuned"

    def __init__(self, gate: Optional[LayaGate] = None) -> None:
        self._gate = gate or LayaGate()

    # -- lifecycle ------------------------------------------------------
    @property
    def gate(self) -> LayaGate:
        return self._gate

    # -- routing --------------------------------------------------------
    def route(self, payload: Dict[str, Any]) -> RouterDecision:
        t0 = time.perf_counter()
        steps: Dict[str, float] = {}

        message = str(payload.get("message") or payload.get("text") or "")
        with _clock(steps, "state"):
            state = build_state(
                message,
                assets=_assets(payload),
                active_image_id=payload.get("active_image_id"),
                last_analysis=payload.get("last_analysis"),
                history=_routing_history(payload.get("history")),
            )

        with _clock(steps, "gate_load"):
            available = self._gate.available()

        if not available:
            return self._base_decision(
                t0, steps, confidence=0.0,
                reason=f"gate-unavailable:{self._gate.load_error or 'no checkpoint'}",
                extra={"gate": None, "gate_error": self._gate.load_error})

        with _clock(steps, "gate_trim"):
            state, dropped, tokens = self._fit(state)

        with _clock(steps, "gate_forward"):
            gate = self._gate.classify_state(state)

        if gate is None:
            return self._base_decision(
                t0, steps, confidence=0.0,
                reason=f"gate-unavailable:{self._gate.load_error or 'predict failed'}",
                extra={"gate": None, "gate_error": self._gate.load_error,
                       "routing_input": state, "input_tokens": tokens})

        extra = {"gate": {k: v for k, v in gate.items() if k != "input"},
                 "router_input": gate.get("input"),
                 "input_tokens": gate.get("input_tokens"),
                 "input_budget": self._gate.state_budget(),
                 "history_turns": len(state["history"]),
                 "history_dropped": dropped,
                 "assets": state["assets"],
                 "active_image_id": state["active_image_id"],
                 "last_analysis": state["last_analysis"]}

        if gate["route"] == "DEEPFAKE":
            if not supported_asset_exists(state):
                # The head asked for forensics, but there is no executable
                # image attached: answer with the base model instead of
                # pretending (or loading a 6 GB pipeline for nothing).
                return self._base_decision(
                    t0, steps, confidence=float(gate["probability"]),
                    reason="gate:deepfake-no-supported-image",
                    extra={**extra, "gate_wanted": "DEEPFAKE"})
            return RouterDecision(
                router=self.name, task="deepfake",
                confidence=float(gate["probability"]),
                reason="gate:finetuned-deepfake",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                steps=steps, abstained=False, extra=extra)

        return RouterDecision(
            router=self.name, task="general",
            confidence=float(gate["probability"]),
            reason="gate:finetuned-base",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            steps=steps, abstained=False, extra=extra)

    # -- internals ------------------------------------------------------
    def routing_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """The exact state that would be sent to the checkpoint (for traces)."""
        return build_state(
            str(payload.get("message") or payload.get("text") or ""),
            assets=_assets(payload),
            active_image_id=payload.get("active_image_id"),
            last_analysis=payload.get("last_analysis"),
            history=_routing_history(payload.get("history")),
        )

    def _fit(self, state: Dict[str, Any]) -> tuple:
        """Drop oldest turns until the state fits; return (state, dropped, tokens).

        The message is never trimmed: `serialize` puts it first, so any residual
        overflow is LAYA's own tail truncation of history, never the request.
        """
        budget = self._gate.state_budget()
        dropped = 0
        tokens = self._gate.token_count(serialize(state))
        while (tokens is not None and tokens > budget and state["history"]):
            state["history"].pop(0)
            dropped += 1
            tokens = self._gate.token_count(serialize(state))
        return state, dropped, tokens

    def _base_decision(self, t0: float, steps: Dict[str, float], *,
                       confidence: float, reason: str,
                       extra: Dict[str, Any]) -> RouterDecision:
        return RouterDecision(
            router=self.name, task="general", confidence=confidence,
            reason=reason, latency_ms=(time.perf_counter() - t0) * 1000.0,
            steps=steps, abstained=False, extra=extra)


class _clock:
    """Tiny per-step stopwatch matching the existing RouterDecision.steps."""

    def __init__(self, sink: Dict[str, float], name: str) -> None:
        self._sink = sink
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.sink_add(time.perf_counter() - self._t0)

    def sink_add(self, seconds: float) -> None:
        self._sink[self._name] = seconds * 1000.0


def _assets(payload: Dict[str, Any]) -> list:
    assets = payload.get("assets")
    if isinstance(assets, list):
        return [a for a in assets if isinstance(a, dict)]
    # Back-compat: an id-only attachment list still routes, as unavailable.
    ids = payload.get("image_ids") or []
    return [{"id": str(i), "kind": "image", "availability": "missing",
             "support": "unsupported"} for i in ids if i]


def _routing_history(history: Any) -> list:
    if not isinstance(history, list):
        return []
    turns = []
    for turn in history[-_MAX_HISTORY_TURNS:]:
        if isinstance(turn, dict) and turn.get("content"):
            turns.append({"role": str(turn.get("role", "user")),
                          "content": str(turn["content"])[:_MAX_TURN_CHARS]})
    return turns
