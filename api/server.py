"""Dual-routing API: separate access per router, token-streamed answers.

Endpoints:
  POST /route/static/predict  — static router -> adapter -> SSE token stream
  POST /route/laya/predict    — laya router (standalone, NO static fallback) -> SSE
  POST /route/compare         — both routers on the same payload (no generation):
                               decisions + per-step latencies + agreement
  GET  /route/stats           — rolling per-router counters (research)
  GET  /health                 — service health

SSE stream shape (text/event-stream, one JSON per `data:` line):
  {"event": "routing_receipt", ...RouterDecision.to_dict()}
  {"event": "selection_receipt", ...}          # adapter hot-swap timing
  {"event": "token", "index": i, "text": tok,  # flushed as each token arrives
   "latency_ms": ms-this-token, "elapsed_ms": ms-since-gen-start}
  {"event": "generation_receipt", ...}         # prefill/decode/tok-per-s/per-token
  {"event": "done"}

The final answer is NEVER printed as one buffered string: clients render the
concatenated `token` events as they arrive.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import AdapterService, LayaRouter, StaticRouter

logger = logging.getLogger(__name__)
app = FastAPI(title="Dual-Route Forensic Service", version="0.1.0")

_REPO = Path(__file__).resolve().parent.parent
app.mount("/ui", StaticFiles(directory=str(_REPO / "ui"), html=True),
          name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")
_STATIC = StaticRouter()
_LAYA: Optional[LayaRouter] = None
_SERVICE: Optional[AdapterService] = None

_STATS: Dict[str, Dict[str, float]] = {
    "static": {"n": 0, "routing_ms": 0.0},
    "laya": {"n": 0, "routing_ms": 0.0},
    "compare": {"n": 0, "agree": 0},
}


def get_laya() -> LayaRouter:
    global _LAYA
    if _LAYA is None:
        _LAYA = LayaRouter()
    return _LAYA


def get_service() -> AdapterService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AdapterService(
            base_dir=os.getenv("FINAL_BASE_DIR",
                               str(_REPO / "models" / "internvl3_2b_lr_fused")),
            anomaly_dir=os.getenv("FINAL_ANOMALY_DIR",
                                  str(_REPO / "adapters" / "Anomaly_adapter")),
            deepfake_llm_dir=os.getenv(
                "FINAL_DEEPFAKE_LLM_DIR",
                str(_REPO / "adapters" / "Deepfake_adapter" / "llm_adapter")),
            deepfake_vision_dir=os.getenv(
                "FINAL_DEEPFAKE_VISION_DIR",
                str(_REPO / "adapters" / "Deepfake_adapter" / "vision_adapter")),
            custom_heads_path=os.getenv(
                "FINAL_CUSTOM_HEADS",
                str(_REPO / "adapters" / "Deepfake_adapter" / "custom_heads"
                    / "custom_heads.safetensors")),
        )
    return _SERVICE


class PredictRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.0
    stub: Optional[bool] = None  # force stub stream (research/offline runs)


def _sse(event: Dict[str, Any]) -> str:
    return "data: " + json.dumps(event) + "\n\n"


def _predict_stream(router_name: str, req: PredictRequest):
    router = _STATIC if router_name == "static" else get_laya()
    service = get_service()

    decision = router.route(req.payload)
    st = _STATS[router_name]
    st["n"] += 1
    st["routing_ms"] += decision.latency_ms
    yield _sse({"event": "routing_receipt", **decision.to_dict()})

    if decision.abstained:
        yield _sse({"event": "done", "reason": "router-abstained"})
        return

    try:
        selection = service.select(decision.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if selection.get("status") == 501:
        yield _sse({"event": "selection_receipt", **selection})
        yield _sse({"event": "done", "reason": "adapter-unavailable"})
        return
    yield _sse({"event": "selection_receipt", **selection})

    # Tokens flushed as they arrive — never one buffered print.
    for chunk in service.stream(req.prompt or req.payload.get("text", ""),
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                stub=req.stub):
        yield _sse(chunk)
    yield _sse({"event": "done"})


@app.post("/route/static/predict")
def static_predict(req: PredictRequest):
    return StreamingResponse(_predict_stream("static", req),
                             media_type="text/event-stream")


@app.post("/route/laya/predict")
def laya_predict(req: PredictRequest):
    return StreamingResponse(_predict_stream("laya", req),
                             media_type="text/event-stream")


@app.post("/route/compare")
def compare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run BOTH routers on the same payload (no generation)."""
    s = _STATIC.route(payload)
    l = get_laya().route(payload)
    _STATS["static"]["n"] += 1
    _STATS["static"]["routing_ms"] += s.latency_ms
    _STATS["laya"]["n"] += 1
    _STATS["laya"]["routing_ms"] += l.latency_ms
    agree = (s.task == l.task)
    _STATS["compare"]["n"] += 1
    _STATS["compare"]["agree"] += 1 if agree else 0
    return {
        "static": s.to_dict(),
        "laya": l.to_dict(),
        "agreement": agree,
    }


@app.get("/route/stats")
def stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("static", "laya"):
        n = _STATS[name]["n"]
        out[name] = {"n": int(n),
                     "mean_routing_ms": round(_STATS[name]["routing_ms"] / max(n, 1), 3)}
    n = _STATS["compare"]["n"]
    out["compare"] = {"n": int(n),
                      "agreement_rate": round(_STATS["compare"]["agree"] / max(n, 1), 4)}
    return out


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "stub_generation": os.getenv("FINAL_STUB_GENERATION", "0")}
