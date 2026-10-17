"""Laya-routed API: one decision model, token-streamed answers.

Laya is the sole router (the rule-based static router was removed by
design). It picks either a specialist adapter or the plain base model.

Endpoints:
  POST /route/laya/predict    — Laya router (standalone, no substitution) -> SSE
  GET  /route/stats           — rolling Laya routing counters (research)
  GET  /health                — service health

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
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import AdapterService, LayaRouter
from router.adapter_service import TagStripper

logger = logging.getLogger(__name__)
app = FastAPI(title="Laya Forensic Service", version="0.2.0")

_REPO = Path(__file__).resolve().parent.parent
app.mount("/ui", StaticFiles(directory=str(_REPO / "ui"), html=True),
          name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/chat")


@app.get("/chat", include_in_schema=False)
def chat_page():
    return FileResponse(str(_REPO / "ui" / "chat.html"))
_LAYA: Optional[LayaRouter] = None
_SERVICE: Optional[AdapterService] = None

_STATS: Dict[str, Dict[str, float]] = {
    "laya": {"n": 0, "routing_ms": 0.0},
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
                str(_REPO / "adapters" / "Deepfake_adapter" / "llm_adapter_toplevel")),
            deepfake_vision_dir=os.getenv(
                "FINAL_DEEPFAKE_VISION_DIR",
                str(_REPO / "adapters" / "Deepfake_adapter" / "vision_adapter_toplevel")),
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
    repetition_penalty: float = 1.0  # training-faithful default
    stub: Optional[bool] = None  # force stub stream (research/offline runs)


def _sse(event: Dict[str, Any]) -> str:
    return "data: " + json.dumps(event) + "\n\n"


def _predict_stream(req: PredictRequest):
    service = get_service()

    decision = get_laya().route(req.payload)
    st = _STATS["laya"]
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

    # Deepfake runs the dedicated forensic wrapper (classifier verdict +
    # streamed reasoning) and needs pixels — never bare-text generation.
    if decision.task == "deepfake":
        import base64 as _b64
        b64 = req.payload.get("image_base64")
        if not b64:
            yield _sse({"event": "notice",
                        "text": "Deepfake analysis needs an image — "
                                "attach one and ask again."})
            yield _sse({"event": "done", "reason": "no-image"})
            return
        for chunk in service.analyze_image_stream(
                _b64.b64decode(b64), max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty, stub=req.stub):
            yield _sse(chunk)
        yield _sse({"event": "done"})
        return

    # Tokens flushed as they arrive — never one buffered print.
    # Prompt is rebuilt in the adapter's training-time framing (system +
    # chat template); bare user text makes the LR-fused base emit rule traces.
    # The LR-fused base also leaks template closers under this prompt, so the
    # stream passes through the XML-tag filter (tags dropped, text kept).
    task_prompt = service.format_prompt(
        decision.task, [], req.prompt or req.payload.get("text", ""))
    scrub = TagStripper()
    for chunk in service.stream(task_prompt,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                repetition_penalty=req.repetition_penalty,
                                stub=req.stub):
        if chunk.get("event") == "token":
            text = scrub.feed(chunk.get("text", ""))
            if not text:
                continue
            yield _sse({**chunk, "text": text})
        else:
            yield _sse(chunk)
    tail = scrub.flush()
    if tail:
        yield _sse({"event": "token", "text": tail})
    yield _sse({"event": "done"})


@app.post("/route/laya/predict")
def laya_predict(req: PredictRequest):
    return StreamingResponse(_predict_stream(req),
                             media_type="text/event-stream")


@app.get("/route/stats")
def stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = _STATS["laya"]["n"]
    out["laya"] = {"n": int(n),
                   "mean_routing_ms": round(
                       _STATS["laya"]["routing_ms"] / max(n, 1), 3)}
    return out


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "stub_generation": os.getenv("FINAL_STUB_GENERATION", "0")}


# ---------------------------------------------------------------------
# End-user chat: Laya-only routing, internals hidden.
# The user types like ChatGPT; routing + latency stay server-side
# (appended to logs/chat_runs.jsonl for research). The client only ever
# sees: notice (optional) -> token... -> done. No receipts, no timings.
# ---------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list = Field(default_factory=list)  # [{role, content}], last 6 used
    image_base64: Optional[str] = None  # attached image -> deepfake path
    max_new_tokens: int = 128
    temperature: float = 0.0
    repetition_penalty: float = 1.1  # mild loop-breaker for chat
    stub: Optional[bool] = None


_CHAT_ABSTAIN_TEXT = (
    "I couldn't tell which specialist your question needs. "
    "Could you rephrase it — for example, mention whether it's about "
    "a face image, sensor readings, or investments?"
)
_CHAT_UNAVAILABLE_TEXT = (
    "That specialist is still being set up. Please try a question about "
    "face-image forensics or sensor anomalies for now."
)


def _log_chat_run(record: Dict[str, Any]) -> None:
    try:
        logdir = _REPO / "logs"
        logdir.mkdir(exist_ok=True)
        with open(logdir / "chat_runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**record, "ts": time.time()}) + "\n")
    except Exception as exc:  # logging must never break chat
        logger.warning("chat log failed: %s", exc)


def _chat_stream(req: ChatRequest):
    service = get_service()
    payload: Dict[str, Any] = {"text": req.message}
    if req.image_base64:
        payload["image_base64"] = req.image_base64

    # Laya only — it is the sole router by design.
    decision = get_laya().route(payload)
    _STATS["laya"]["n"] += 1
    _STATS["laya"]["routing_ms"] += decision.latency_ms

    log_record: Dict[str, Any] = {
        "message": req.message,
        "routing": decision.to_dict(),
    }

    if decision.abstained:
        _log_chat_run({**log_record, "outcome": "abstained"})
        yield _sse({"event": "notice", "text": _CHAT_ABSTAIN_TEXT})
        yield _sse({"event": "done"})
        return

    try:
        selection = service.select(decision.task)
    except ValueError:
        _log_chat_run({**log_record, "outcome": "bad-task"})
        yield _sse({"event": "notice", "text": _CHAT_ABSTAIN_TEXT})
        yield _sse({"event": "done"})
        return
    if selection.get("status") == 501:
        _log_chat_run({**log_record, "outcome": "adapter-unavailable",
                       "selection": selection})
        yield _sse({"event": "notice", "text": _CHAT_UNAVAILABLE_TEXT})
        yield _sse({"event": "done"})
        return
    log_record["selection"] = selection

    # The forensic judge needs the pixels: text-only deepfake requests get a
    # friendly ask-for-image instead of a base-model confabulation.
    if decision.task == "deepfake" and "image_base64" not in payload:
        _log_chat_run({**log_record, "outcome": "no-image"})
        yield _sse({"event": "notice",
                    "text": "Please attach the face image using the 📎 button "
                            "and I'll examine whether it's real or fake."})
        yield _sse({"event": "done"})
        return

    # Deepfake runs the dedicated forensic wrapper (verdict + streamed
    # reasoning) and needs pixels — handled before the text-token path below.
    if decision.task == "deepfake":
        import base64 as _b64
        gen_receipt = {}
        for chunk in service.analyze_image_stream(
                _b64.b64decode(payload["image_base64"]),
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty, stub=req.stub):
            if chunk.get("event") == "verdict":
                yield _sse({"event": "verdict", **{k: v for k, v in chunk.items()
                                                   if k != "event"}})
                log_record["verdict"] = chunk
            elif chunk.get("event") == "token":
                # analyze_image_stream already drops the trained XML tags
                # (<fast>/<planning>/<reasoning>/<answer>…) and keeps the text.
                yield _sse({"event": "token", "text": chunk.get("text", "")})
            elif chunk.get("event") == "notice":
                yield _sse({"event": "notice", "text": chunk.get("text", "")})
            elif chunk.get("event") == "generation_receipt":
                gen_receipt = chunk
        _log_chat_run({**log_record, "outcome": "answered",
                       "generation": gen_receipt})
        yield _sse({"event": "done"})
        return

    # Quiet context: last turns travel as proper chat turns; the prompt is
    # rebuilt in the adapter's training-time framing (task system prompt +
    # fused chat template). Bare user text makes the LR-fused base fall back
    # to logic-rule traces.
    norm_turns = []
    for t in (req.history or [])[-6:]:
        if isinstance(t, ChatTurn):
            t = t.dict()
        if isinstance(t, dict) and t.get("content"):
            norm_turns.append({"role": t.get("role", "user"),
                               "content": t.get("content")})
    prompt = service.format_prompt(decision.task, norm_turns, req.message)

    # Tokens only — latencies stay in the server log, never on the wire.
    # Every task runs the streaming XML-tag filter: tags are dropped, the
    # reasoning text is kept (the LR-fused base leaks template closers, and
    # the forensic LM emits <fast>/<planning>/…/<answer> by training design).
    scrub = TagStripper()
    gen_receipt: Dict[str, Any] = {}
    for chunk in service.stream(prompt,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                repetition_penalty=req.repetition_penalty,
                                stub=req.stub):
        if chunk.get("event") == "token":
            text = scrub.feed(chunk.get("text", ""))
            if not text:
                continue
            yield _sse({"event": "token", "text": text})
        elif chunk.get("event") == "generation_receipt":
            gen_receipt = chunk
    tail = scrub.flush()
    if tail:
        yield _sse({"event": "token", "text": tail})
    _log_chat_run({**log_record, "outcome": "answered",
                   "generation": gen_receipt})
    yield _sse({"event": "done"})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(_chat_stream(req),
                             media_type="text/event-stream")
