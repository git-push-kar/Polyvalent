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
# A single-turn pass may overrule BASE only on an unambiguous DEEPFAKE.
_SINGLE_TURN_FLOOR = 0.90
# Re-running forensics on an image that is not attached is a bigger commitment
# than answering from the saved analysis, so the contextual pass must be
# near-certain. Measured on the same thread: "check it again" 0.992,
# "is this real or fake" 0.994, greeting 0.64-0.86.
_UNATTACHED_FLOOR = 0.95


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

        promoted = None
        if gate["route"] == "BASE":
            # History can hold the conversation on the previous topic (a
            # description turn keeps "do i look real" on BASE even with the
            # image present). The same message scored on the image alone is
            # the checkpoint's own view of the request, so a confident
            # DEEPFAKE there wins. This only promotes; it never demotes.
            promoted = self._single_turn(payload, steps)
            if promoted is not None:
                gate = promoted["gate"]
                extra["gate"] = {k: v for k, v in gate.items() if k != "input"}
                extra["router_input"] = gate.get("input")
                extra["input_tokens"] = gate.get("input_tokens")
                extra["single_turn_gate"] = promoted["view"]
                extra["promotion"] = promoted["reason"]

        if (gate["route"] == "DEEPFAKE" and promoted is None
                and not payload.get("attached_now")
                and float(gate["probability"]) < _UNATTACHED_FLOOR):
            # An image from earlier in the thread, a message that is not an
            # explicit request: answer from the saved state rather than
            # loading the pipeline on a history-inflated signal.
            extra["suppressed_deepfake"] = gate["probability"]
            extra["base_reason"] = "gate:deepfake-below-unattached-floor"
            gate = {**gate, "route": "BASE",
                    "probability": round(1.0 - float(gate["probability"]), 4)}
            extra["gate"] = {k: v for k, v in gate.items() if k != "input"}

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
                reason=str(extra.get("promotion") or "gate:finetuned-deepfake"),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                steps=steps, abstained=False, extra=extra)

        return RouterDecision(
            router=self.name, task="general",
            confidence=float(gate["probability"]),
            reason=str(extra.pop("base_reason", "gate:finetuned-base")),
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

    def _single_turn(self, payload: Dict[str, Any],
                     steps: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Score the message against the image with no history, if one is near.

        Runs only when the image is attached to this request or arrived within
        the recent-message window, so an old image cannot resurrect DEEPFAKE.
        """
        if not (payload.get("attached_now") or payload.get("recent_image_id")):
            return None
        state = build_state(
            str(payload.get("message") or payload.get("text") or ""),
            assets=_assets(payload),
            active_image_id=payload.get("active_image_id"),
            last_analysis=payload.get("last_analysis"),
            history=[],
        )
        if not supported_asset_exists(state):
            return None
        state, dropped, tokens = self._fit(state)
        with _clock(steps, "gate_single_turn"):
            gate = self._gate.classify_state(state)
        if gate is None or gate["route"] != "DEEPFAKE":
            return None
        if float(gate["probability"]) < _SINGLE_TURN_FLOOR:
            return None
        return {
            "gate": gate,
            "view": {k: v for k, v in gate.items() if k != "input"},
            "reason": "gate:single-turn-deepfake",
        }

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
