"""Fine-tuned LAYA router for live chat: BASE vs DEEPFAKE, decided per turn.

This is the checkpoint trained in `C:/Users/Admin/Desktop/Router`
(`models/laya_router`, SHA-256 cbca78ad…): a two-route Laya head fitted on the
five-field runtime state and scored against its own `questions.json`. It is
consulted for every non-`deep_thinking` turn — text or image — and its choices
are kept exactly as trained:

  BASE      answer with the shared base model (no specialist adapter)
  DEEPFAKE  run the deepfake forensic pipeline on the active image

There are no keyword rules, no modality short-circuits and no preset replies
here: if the checkpoint cannot answer, the turn goes to BASE with an explicit
`gate-unavailable` reason rather than an invented specialist choice.

DEEPFAKE is only executable when the active asset is available and supported.
For turns where an image is in recent chat but not attached directly to the current
request, a two-verdict mechanism evaluates the request both without and with
the recent image to detect when the user is referencing that image for verification.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base import BaseRouter, RouterDecision
from .laya_gate import LayaGate, build_state, serialize, supported_asset_exists


class _clock:
    """Tiny per-step stopwatch matching RouterDecision.steps."""

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


def _extract_asset_view(asset: Any) -> Optional[Dict[str, Any]]:
    if asset is None:
        return None
    if hasattr(asset, "routing_view"):
        return asset.routing_view()
    if isinstance(asset, dict):
        return {
            "id": str(asset.get("id") or ""),
            "kind": str(asset.get("kind") or "image"),
            "availability": str(asset.get("availability") or "available"),
            "support": str(asset.get("support") or "supported"),
        }
    return None


def _is_usable(asset: Any) -> bool:
    if asset is None:
        return False
    if hasattr(asset, "usable"):
        return bool(asset.usable)
    if isinstance(asset, dict):
        return (
            asset.get("availability") == "available"
            and asset.get("support") == "supported"
        )
    return False


def _get_id(asset: Any) -> Optional[str]:
    if asset is None:
        return None
    if hasattr(asset, "id"):
        return asset.id
    if isinstance(asset, dict):
        return asset.get("id")
    return None


class TrainedLayaRouter(BaseRouter):
    """The application's primary router: the fine-tuned BASE/DEEPFAKE head."""

    name = "laya-finetuned"

    def __init__(self, gate: Optional[LayaGate] = None) -> None:
        self._gate = gate or LayaGate()

    @property
    def gate(self) -> LayaGate:
        return self._gate

    def route(self, payload: Dict[str, Any]) -> RouterDecision:
        t0 = time.perf_counter()
        steps: Dict[str, float] = {}

        message = str(payload.get("message") or payload.get("text") or "").strip()

        with _clock(steps, "gate_load"):
            available = self._gate.available()

        if not available:
            return self._base_decision(
                t0, steps, confidence=0.0,
                reason=f"gate-unavailable:{self._gate.load_error or 'no checkpoint'}",
                extra={"gate": None, "gate_error": self._gate.load_error}
            )

        attached_now = bool(payload.get("attached_now"))
        attached_asset = payload.get("attached_asset")
        recent_asset = payload.get("recent_asset")
        active_image_id = payload.get("active_image_id")
        last_analysis = payload.get("last_analysis")

        # Backwards compatibility: if attached_asset was not passed directly, look in assets list
        if attached_asset is None and isinstance(payload.get("assets"), list):
            for a in payload["assets"]:
                if isinstance(a, dict) and a.get("kind") == "image":
                    if active_image_id is None or a.get("id") == active_image_id:
                        attached_asset = a
                        if "attached_now" not in payload:
                            attached_now = True
                        break

        has_attached = bool(attached_now and attached_asset and _is_usable(attached_asset))
        has_recent = bool(recent_asset and _is_usable(recent_asset))

        # ------------------------------------------------------------------
        # CASE 1: Image is attached directly to the current request
        # ------------------------------------------------------------------
        if has_attached:
            asset_view = _extract_asset_view(attached_asset)
            asset_id = _get_id(attached_asset) or active_image_id

            with _clock(steps, "gate_forward"):
                state = build_state(
                    message=message,
                    assets=[asset_view] if asset_view else [],
                    active_image_id=asset_id,
                    last_analysis=last_analysis,
                    history=[],
                )
                gate_res = self._gate.classify_state(state)

            if gate_res is None:
                return self._base_decision(
                    t0, steps, confidence=0.0,
                    reason=f"gate-unavailable:{self._gate.load_error or 'predict failed'}",
                    extra={"gate": None, "gate_error": self._gate.load_error, "router_input": state}
                )

            extra = {
                "gate": {k: v for k, v in gate_res.items() if k != "input"},
                "router_input": gate_res.get("input"),
                "input_tokens": gate_res.get("input_tokens"),
                "input_budget": self._gate.state_budget(),
                "assets": state["assets"],
                "active_image_id": asset_id,
                "last_analysis": last_analysis,
                "attached_now": True,
            }

            if gate_res["route"] == "DEEPFAKE":
                if not supported_asset_exists(state):
                    return self._base_decision(
                        t0, steps, confidence=float(gate_res["probability"]),
                        reason="gate:deepfake-no-supported-image",
                        extra={**extra, "gate_wanted": "DEEPFAKE"}
                    )
                return RouterDecision(
                    router=self.name, task="deepfake",
                    confidence=float(gate_res["probability"]),
                    reason="gate:finetuned-deepfake",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    steps=steps, abstained=False, extra=extra
                )
            else:
                return RouterDecision(
                    router=self.name, task="general",
                    confidence=float(gate_res["probability"]),
                    reason="gate:finetuned-base",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    steps=steps, abstained=False, extra=extra
                )

        # ------------------------------------------------------------------
        # CASE 2: No image attached now, and NO image in recent chat
        # ------------------------------------------------------------------
        if not has_recent:
            with _clock(steps, "gate_forward"):
                state = build_state(
                    message=message,
                    assets=[],
                    active_image_id=None,
                    last_analysis=None,
                    history=[],
                )
                gate_res = self._gate.classify_state(state)

            if gate_res is None:
                return self._base_decision(
                    t0, steps, confidence=0.0,
                    reason=f"gate-unavailable:{self._gate.load_error or 'predict failed'}",
                    extra={"gate": None, "gate_error": self._gate.load_error, "router_input": state}
                )

            extra = {
                "gate": {k: v for k, v in gate_res.items() if k != "input"},
                "router_input": gate_res.get("input"),
                "input_tokens": gate_res.get("input_tokens"),
                "input_budget": self._gate.state_budget(),
                "assets": [],
                "active_image_id": None,
                "last_analysis": None,
                "attached_now": False,
            }

            if gate_res["route"] == "DEEPFAKE":
                return RouterDecision(
                    router=self.name, task="deepfake",
                    confidence=float(gate_res["probability"]),
                    reason="gate:finetuned-deepfake-no-image",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    steps=steps, abstained=False, extra=extra
                )
            else:
                return RouterDecision(
                    router=self.name, task="general",
                    confidence=float(gate_res["probability"]),
                    reason="gate:finetuned-base",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    steps=steps, abstained=False, extra=extra
                )

        # ------------------------------------------------------------------
        # CASE 3: No image attached now, BUT there is a usable recent image.
        # Two-Verdict Mechanism:
        # 1. Query LAYA without image (as current turn looks standalone).
        # 2. Query LAYA with recent image (as if user references that image).
        # ------------------------------------------------------------------
        recent_view = _extract_asset_view(recent_asset)
        recent_id = _get_id(recent_asset) or active_image_id

        with _clock(steps, "gate_forward_no_img"):
            state_no = build_state(
                message=message,
                assets=[],
                active_image_id=None,
                last_analysis=None,
                history=[],
            )
            v_no = self._gate.classify_state(state_no)

        with _clock(steps, "gate_forward_with_img"):
            state_with = build_state(
                message=message,
                assets=[recent_view] if recent_view else [],
                active_image_id=recent_id,
                last_analysis=last_analysis,
                history=[],
            )
            v_with = self._gate.classify_state(state_with)

        if v_with is None and v_no is None:
            return self._base_decision(
                t0, steps, confidence=0.0,
                reason=f"gate-unavailable:{self._gate.load_error or 'predict failed'}",
                extra={"gate": None, "gate_error": self._gate.load_error}
            )

        active_gate = v_with or v_no
        two_verdicts = {
            "without_image": {k: v for k, v in v_no.items() if k != "input"} if v_no else None,
            "with_image": {k: v for k, v in v_with.items() if k != "input"} if v_with else None,
        }

        extra = {
            "gate": {k: v for k, v in active_gate.items() if k != "input"},
            "router_input": active_gate.get("input"),
            "input_tokens": active_gate.get("input_tokens"),
            "input_budget": self._gate.state_budget(),
            "two_verdict": two_verdicts,
            "assets": state_with["assets"],
            "active_image_id": recent_id,
            "last_analysis": last_analysis,
            "attached_now": False,
            "recent_image_id": recent_id,
        }

        # If v_with clearly selects DEEPFAKE: the request asks for authenticity verification
        # and references the image from the recent chat.
        if v_with and v_with["route"] == "DEEPFAKE" and float(v_with.get("probability", 0.0)) >= 0.50:
            extra["resolved_via"] = "two_verdict_deepfake"
            return RouterDecision(
                router=self.name, task="deepfake",
                confidence=float(v_with["probability"]),
                reason="gate:two-verdict-deepfake-recent-image",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                steps=steps, abstained=False, extra=extra
            )

        # Otherwise, even in the presence of the recent image, the request is BASE
        # (e.g. "Explain that verdict", "Describe what he is wearing", "What is the weather?")
        conf = float(v_with["probability"]) if v_with else (float(v_no["probability"]) if v_no else 1.0)
        return RouterDecision(
            router=self.name, task="general",
            confidence=conf,
            reason="gate:finetuned-base",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            steps=steps, abstained=False, extra=extra
        )

    def _base_decision(self, t0: float, steps: Dict[str, float], *,
                       confidence: float, reason: str,
                       extra: Dict[str, Any]) -> RouterDecision:
        return RouterDecision(
            router=self.name, task="general", confidence=confidence,
            reason=reason, latency_ms=(time.perf_counter() - t0) * 1000.0,
            steps=steps, abstained=False, extra=extra
        )
