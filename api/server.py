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
  {"event": "thinking", "text": tok}           # deepfake reasoning (collapsed)
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
from router.adapter_service import TagSectioner, TagStripper
from router.prompts import clean_adapter_directive

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


_PRISTINE_BASE = r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B"
_V2_ADAPTERS = (r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\runs"
                r"\intern_v2_ground\adapters")


def get_service() -> AdapterService:
    """One service, one pristine base, three task adapters.

    All paths default to the training repositories and are env-overridable,
    so the weights keep their provenance instead of being copied around.
    """
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AdapterService(
            base_dir=os.getenv("FINAL_BASE_DIR", _PRISTINE_BASE),
            anomaly_dir=os.getenv(
                "FINAL_ANOMALY_DIR",
                r"C:\Users\Admin\Desktop\anomaly\outputs\tep_rca_internvl3_2b"
                r"\internvl3_2b_pristine_rca"),
            reasoning_dir=os.getenv(
                "FINAL_REASONING_DIR",
                r"C:\Users\Admin\Desktop\C68-LR\models_v3\adapter\adapter"),
            deepfake_llm_dir=os.getenv(
                "FINAL_DEEPFAKE_LLM_DIR", _V2_ADAPTERS + r"\llm_adapter"),
            deepfake_vision_dir=os.getenv(
                "FINAL_DEEPFAKE_VISION_DIR", _V2_ADAPTERS + r"\vision_adapter"),
            custom_heads_path=os.getenv(
                "FINAL_CUSTOM_HEADS",
                _V2_ADAPTERS + r"\custom_heads\custom_heads.safetensors"),
        )
    return _SERVICE


class PredictRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    max_new_tokens: int = 512
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
    # Reasoning emits <reasoning>...</reasoning><answer>...</answer>: section
    # the first into `thinking` (collapsed) and the second into `token`.
    # Other tasks only need tag stripping.
    sectioner = (TagSectioner(default="answer")
                 if decision.task == "reasoning" else None)
    scrub = TagStripper()
    saw_visible = False
    for chunk in service.stream(task_prompt,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                repetition_penalty=req.repetition_penalty,
                                stub=req.stub):
        if chunk.get("event") == "token":
            raw = chunk.get("text", "")
            if sectioner is not None:
                for channel, text in sectioner.feed(raw):
                    if not text:
                        continue
                    if channel != "thinking":
                        saw_visible = True
                    yield _sse({**chunk,
                                "event": ("thinking" if channel == "thinking"
                                          else "token"),
                                "text": text})
            else:
                text = scrub.feed(raw)
                if not text:
                    continue
                yield _sse({**chunk, "text": text})
        else:
            yield _sse(chunk)
    if sectioner is not None:
        for channel, text in sectioner.flush():
            if text:
                if channel != "thinking":
                    saw_visible = True
                yield _sse({"event": ("thinking" if channel == "thinking"
                                      else "token"), "text": text})
        if not saw_visible:
            fallback = "".join(sectioner.thinking_text).strip()
            if fallback:
                yield _sse({"event": "token", "text": fallback})
    else:
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
    max_new_tokens: int = 768  # room for fast+planning+reasoning+conclusion+answer
    temperature: float = 0.0
    repetition_penalty: float = 1.1  # mild loop-breaker for chat
    stub: Optional[bool] = None


_CHAT_ABSTAIN_TEXT = (
    "I couldn't tell which specialist your question needs. "
    "Could you rephrase it — for example, mention whether it's about "
    "a face image, sensor readings, or a logic/reasoning problem?"
)
_CHAT_UNAVAILABLE_TEXT = (
    "That specialist is still being set up. Please try a question about "
    "face-image forensics, sensor anomalies, or logical reasoning for now."
)
# Canned notices are UI feedback, not dialogue. Clients (including cached
# older pages) may echo them back in `history`; drop them so the base model
# cannot pattern-match and repeat the notice on the next turn.
_HISTORY_NOISE = (
    "I couldn't tell which specialist",
    "That specialist is still being set up",
    "Please attach the face image",
    "I can check a face image for you",
    "That file doesn't look like an image",
    "Reasoning hit an error",
    "Generation hit an error",
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

    # The forensic judge needs pixels: ask for the image BEFORE selecting /
    # loading the 6 GB deepfake wrapper, so text-only requests stay instant.
    if decision.task == "deepfake" and "image_base64" not in payload:
        _log_chat_run({**log_record, "outcome": "no-image"})
        yield _sse({"event": "notice",
                    "text": "I can check a face image for you — attach it with "
                            "the 📎 button and ask again. For anything else, "
                            "just type your question."})
        yield _sse({"event": "done"})
        return

    # Chat UX: an abstention is not a wall. With no specialist signal the
    # plain base model answers; the raw Laya decision stays in the log.
    if decision.task == "needs_clarification":
        _log_chat_run({**log_record, "outcome": "abstain->general"})
        decision.task = "general"
        decision.confidence = 0.0
        decision.reason = "guard:general"
        decision.abstained = False

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
            elif chunk.get("event") == "thinking":
                # analyze_image_stream routes the reasoning sections here
                # (<fast>/<planning>/<reasoning>/<reflection>/<conclusion>);
                # the client collapses them under a "Reasoning" disclosure.
                yield _sse({"event": "thinking", "text": chunk.get("text", "")})
            elif chunk.get("event") == "token":
                # The <answer> section — the visible forensic reply.
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
        if not (isinstance(t, dict) and t.get("content")):
            continue
        role = t.get("role", "user")
        content = str(t.get("content"))
        if role == "assistant" and content.startswith(_HISTORY_NOISE):
            continue  # retry notices must never become model context
        norm_turns.append({"role": role, "content": content})
    prompt_message = (clean_adapter_directive(req.message)
                      if decision.task == "reasoning" else req.message)
    prompt = service.format_prompt(decision.task, norm_turns, prompt_message)

    # Tokens only — latencies stay in the server log, never on the wire.
    # The reasoning adapter emits <reasoning>...</reasoning><answer>...</answer>:
    # the first section streams as `thinking` (collapsed by the client), the
    # second as visible `token`s. Every other task gets plain tag stripping.
    sectioner = (TagSectioner(default="answer")
                 if decision.task == "reasoning" else None)
    scrub = TagStripper()
    saw_visible = False
    gen_receipt: Dict[str, Any] = {}
    for chunk in service.stream(prompt,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                repetition_penalty=req.repetition_penalty,
                                stub=req.stub):
        if chunk.get("event") == "token":
            raw = chunk.get("text", "")
            if sectioner is not None:
                for channel, text in sectioner.feed(raw):
                    if not text:
                        continue
                    if channel != "thinking":
                        saw_visible = True
                    yield _sse({"event": ("thinking" if channel == "thinking"
                                          else "token"), "text": text})
            else:
                text = scrub.feed(raw)
                if not text:
                    continue
                yield _sse({"event": "token", "text": text})
        elif chunk.get("event") == "generation_receipt":
            gen_receipt = chunk
    if sectioner is not None:
        for channel, text in sectioner.flush():
            if text:
                if channel != "thinking":
                    saw_visible = True
                yield _sse({"event": ("thinking" if channel == "thinking"
                                      else "token"), "text": text})
        if not saw_visible:
            fallback = "".join(sectioner.thinking_text).strip()
            if fallback:
                yield _sse({"event": "token", "text": fallback})
    else:
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
