"""Shared routing contract: both routers return the same decision schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


TASKS = ("anomaly", "deepfake", "portfolio", "general", "needs_clarification")

# Portfolio adapter is still in development: routers may select it, but the
# adapter service answers 501 until weights land.
PORTFOLIO_STUB = True


@dataclass
class RouterDecision:
    router: str  # "static" | "laya"
    task: str
    confidence: float
    reason: str
    latency_ms: float
    steps: Dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "router": self.router,
            "task": self.task,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 3),
            "steps": {k: round(v, 3) for k, v in self.steps.items()},
            "abstained": self.abstained,
            "extra": self.extra,
        }


class BaseRouter:
    name = "base"

    def route(self, payload: Dict[str, Any]) -> RouterDecision:
        raise NotImplementedError
