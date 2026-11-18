"""Fine-tuned Laya router (BASE / DEEPFAKE) trained in the `Router/` project.

The zero-shot `laya.Router` used by `LayaRouter` is a general decision model.
`Router/` retrained a Laya head on the five-field runtime state specifically for
the deepfake boundary and measured 98.0 % accuracy / 0.9787 macro-F1 on its
frozen synthetic test (see `models/laya_router/comparison_provenance.json`).
That checkpoint is a two-route model: DEEPFAKE only when the current request
asks for a *new* authenticity verdict on a resolvable image; everything else is
BASE.

This module wraps the checkpoint as a gate, not as a second full router: it
decides whether the current turn belongs to the deepfake specialist, and
`LayaRouter` keeps ownership of anomaly / reasoning / general. The three files
the checkpoint depends on travel with it:

  model.safetensors   encoder + head (SHA-256 cbca78ad4c27a037... )
  encoder/, tokenizer/
  questions.json      the exact question spec the head was fitted against

`input_serializer.py` is mirrored in `serialize()` below: the archive's contract
is a fixed five-key ordering (message first so the newest request is never lost
behind history), and the model was trained on that byte layout.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_DIR_ENV = "FINAL_LAYA_ROUTER_DIR"
_DEVICE_ENV = "FINAL_LAYA_ROUTER_DEVICE"
_KEYS = ("message", "assets", "active_image_id", "last_analysis", "history")
_ORDER = ("message", "assets", "active_image_id", "last_analysis", "history")
_MAX_LEN = 512
_HEAD_MAX_LEN = 192


def model_dir() -> Optional[Path]:
    """Checkpoint directory, or None when the weights are not on this machine."""
    raw = (os.getenv(_DIR_ENV) or "").strip().strip('"')
    cand = Path(raw).expanduser() if raw else _REPO / "models" / "laya_router"
    if not cand.is_absolute():
        cand = _REPO / cand
    if (cand / "model.safetensors").is_file() and (cand / "questions.json").is_file():
        return cand
    return None


def serialize(state: Dict[str, Any]) -> str:
    """The archive's `input_serializer.serialize`, byte-for-byte in key order."""
    if set(state) != set(_KEYS):
        raise ValueError("Unexpected/missing runtime state keys")
    ordered = {k: state[k] for k in _ORDER}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


class LayaGate:
    """Lazy, cached access to the fine-tuned BASE/DEEPFAKE checkpoint."""

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device or (os.getenv(_DEVICE_ENV) or "cpu")
        self._agent = None
        self._questions: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------
    def _ensure(self) -> bool:
        if self._agent is not None:
            return True
        with self._lock:
            if self._agent is not None:
                return True
            directory = model_dir()
            if directory is None:
                self._load_error = f"checkpoint not found ({_DIR_ENV} unset)"
                return False
            try:
                from . import _torch_shim  # noqa: F401  (Windows-CPU hardening)
                from laya.agent import Agent

                self._questions = json.loads(
                    (directory / "questions.json").read_text(encoding="utf-8"))
                self._agent = Agent(model_id_or_path=str(directory),
                                    device=self._device)
                return True
            except Exception as exc:  # laya/torch missing, corrupt checkpoint ...
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("fine-tuned Laya router unavailable: %s",
                               self._load_error)
                return False

    def available(self) -> bool:
        return self._ensure()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # -- inference -----------------------------------------------------
    def classify(self, message: str, *, image_attached: bool = False,
                 last_analysis: Optional[Dict[str, Any]] = None,
                 history: Optional[list] = None
                 ) -> Optional[Dict[str, Any]]:
        """Return the trained route for this turn, or None if the gate is down.

        `image_attached` mirrors the archive's asset list: a supported,
        available image is what makes DEEPFAKE executable at all.
        """
        if not self._ensure():
            return None
        assets = ([{"id": "active_image", "kind": "image",
                    "availability": "available", "support": "supported"}]
                  if image_attached else [])
        state = {
            "message": message or "",
            "assets": assets,
            "active_image_id": "active_image" if image_attached else None,
            "last_analysis": last_analysis,
            "history": history or [],
        }
        t0 = time.perf_counter()
        try:
            raw = self._agent.predict(serialize(state), self._questions,
                                      max_len=_MAX_LEN,
                                      head_max_len=_HEAD_MAX_LEN)
        except Exception as exc:
            self._load_error = f"predict failed: {type(exc).__name__}: {exc}"
            logger.warning("fine-tuned Laya router predict failed: %s", exc)
            return None
        answer = ((raw.get("answers") or {}).get("route") or {})
        choice = str(answer.get("choice") or "BASE").upper()
        probs = answer.get("probabilities") or {}
        p = answer.get("answer_confidence")
        if not isinstance(p, (int, float)):
            p = probs.get(choice, 0.0) if isinstance(probs, dict) else 0.0
        return {
            "route": "DEEPFAKE" if choice == "DEEPFAKE" else "BASE",
            "probability": round(float(p), 4),
            "probabilities": probs,
            "model": raw.get("model", "laya-router"),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
