"""Router package: static (rules) + laya (System-1) with per-step latency."""
from .adapter_service import AdapterService
from .base import PORTFOLIO_STUB, TASKS, BaseRouter, RouterDecision
from .latency import GenerationReceipt, StepClock, TokenStat
from .laya_router import LayaRouter
from .prompts import SYSTEM_BY_TASK, build_messages
from .static_router import StaticRouter

__all__ = [
    "PORTFOLIO_STUB",
    "TASKS",
    "AdapterService",
    "BaseRouter",
    "GenerationReceipt",
    "LayaRouter",
    "RouterDecision",
    "StaticRouter",
    "StepClock",
    "SYSTEM_BY_TASK",
    "TokenStat",
    "build_messages",
]
