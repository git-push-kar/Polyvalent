"""Router package: Laya decides every turn, with per-step latency.

Chat uses the fine-tuned BASE/DEEPFAKE checkpoint (`TrainedLayaRouter`); the
anomaly and research endpoints keep the four-way zero-shot `LayaRouter`. The
rule-based StaticRouter was removed by design, so latency and accuracy numbers
stay attributable to a decision model rather than to hand-written cues.
"""
from .adapter_service import AdapterService
from .base import TASKS, BaseRouter, RouterDecision
from .conversations import Analysis, ConversationStore
from .latency import GenerationReceipt, StepClock, TokenStat
from .laya_router import LayaRouter
from .prompts import SYSTEM_BY_TASK, build_messages
from .trained_router import TrainedLayaRouter

__all__ = [
    "TASKS",
    "AdapterService",
    "Analysis",
    "BaseRouter",
    "ConversationStore",
    "GenerationReceipt",
    "LayaRouter",
    "RouterDecision",
    "StepClock",
    "SYSTEM_BY_TASK",
    "TokenStat",
    "TrainedLayaRouter",
    "build_messages",
]
