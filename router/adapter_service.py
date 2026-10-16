"""Shared adapter service: one frozen base, named PEFT adapters, hot-swap.

Routing layer picks the task; this service executes it. Generation is a
token-by-token stream (never a single buffered string) so the API can flush
each token to the client as it arrives, with per-token latency attached.

Two generation backends:
  * real  — transformers generate() + TextIteratorStreamer on a thread,
            through PeftModel with adapter_name= switching.
  * stub  — deterministic word-split stream (FINAL_STUB_GENERATION=1 or
            model unloadable); verifies streaming shape without GPU.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from router.latency import GenerationReceipt, TokenStat

logger = logging.getLogger(__name__)

_ANOMALY = "anomaly"
_DEEPFAKE_LLM = "deepfake_llm"
_DEEPFAKE_VISION = "deepfake_vision"


class AdapterService:
    def __init__(self, base_dir: str,
                 anomaly_dir: str,
                 deepfake_llm_dir: str,
                 deepfake_vision_dir: str,
                 custom_heads_path: Optional[str] = None) -> None:
        self.base_dir = base_dir
        self.anomaly_dir = anomaly_dir
        self.deepfake_llm_dir = deepfake_llm_dir
        self.deepfake_vision_dir = deepfake_vision_dir
        self.custom_heads_path = custom_heads_path
        self._model = None
        self._tokenizer = None
        self._active: Optional[str] = None
        self._heads_loaded = False

    # -- adapter selection --------------------------------------------
    def select(self, task: str) -> Dict[str, Any]:
        """Activate the adapter set for `task`. Returns selection receipt."""
        t0 = time.perf_counter()
        if task == "portfolio":
            return {"task": task, "status": 501,
                    "detail": "portfolio adapter in development (stub)",
                    "latency_ms": 0.0}
        try:
            self._ensure_model()
        except Exception as exc:
            if os.getenv("FINAL_STUB_GENERATION", "0") == "1":
                logger.warning("model unavailable, stub selection: %s", exc)
                names = {"anomaly": [_ANOMALY],
                         "deepfake": [_DEEPFAKE_LLM, _DEEPFAKE_VISION,
                                      "custom_heads"],
                         "general": ["base"]}[task]
                return {"task": task, "status": 200, "adapters": names,
                        "mode": "stub",
                        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
            raise
        if task == "anomaly":
            self._model.set_adapter(_ANOMALY)
            self._active = _ANOMALY
            names = [_ANOMALY]
        elif task == "deepfake":
            self._model.set_adapter([_DEEPFAKE_LLM, _DEEPFAKE_VISION])
            self._load_heads_once()
            self._active = "deepfake"
            names = [_DEEPFAKE_LLM, _DEEPFAKE_VISION, "custom_heads"]
        elif task == "general":
            self._model.disable_adapter_layers()
            self._active = None
            names = ["base"]
        else:
            raise ValueError(f"unknown task: {task}")
        return {"task": task, "status": 200, "adapters": names,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_dir, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModel.from_pretrained(
            self.base_dir, trust_remote_code=True,
            torch_dtype="auto", device_map="auto",
        )
        model = PeftModel.from_pretrained(
            base, self.deepfake_llm_dir,
            adapter_name=_DEEPFAKE_LLM, is_trainable=False)
        model.load_adapter(self.deepfake_vision_dir,
                           adapter_name=_DEEPFAKE_VISION)
        model.load_adapter(self.anomaly_dir, adapter_name=_ANOMALY)
        model.eval()
        self._model = model

    def _load_heads_once(self) -> None:
        if self._heads_loaded or not self.custom_heads_path:
            return
        from safetensors.torch import load_file as _load
        heads = _load(self.custom_heads_path, device="cpu")
        base_keys = set(self._model.get_base_model().state_dict().keys())
        overlap = base_keys.intersection(heads.keys())
        assert not overlap, (
            f"custom_heads overwrite backbone keys: {sorted(overlap)[:5]}; "
            "refusing to contaminate shared base (rename to extra modules)")
        self._model.load_state_dict(heads, strict=False)
        self._heads_loaded = True

    # -- streaming generation ------------------------------------------
    def stream(self, prompt: str, max_new_tokens: int = 256,
               temperature: float = 0.0,
               stub: Optional[bool] = None) -> Generator[Dict[str, Any], None, None]:
        """Yield token events then a final generation receipt.

        Event shapes:
          {"event": "token", "index": i, "text": tok,
           "latency_ms": ms-for-this-token, "elapsed_ms": ms-since-start}
          {"event": "generation_receipt", **GenerationReceipt.to_dict()}
        """
        use_stub = (os.getenv("FINAL_STUB_GENERATION", "0") == "1"
                    if stub is None else stub)
        if use_stub or self._model is None and self._stub_required():
            yield from self._stub_stream(prompt, max_new_tokens)
            return
        yield from self._real_stream(prompt, max_new_tokens, temperature)

    def _stub_required(self) -> bool:
        try:
            self._ensure_model()
            return False
        except Exception as exc:  # weights/GPU absent: stub keeps shape testable
            logger.warning("model unavailable, stub stream: %s", exc)
            return True

    def _stub_stream(self, prompt: str,
                     max_new_tokens: int) -> Generator[Dict[str, Any], None, None]:
        words = ("Analysis of the request shows task-relevant evidence. "
                 "Key indicators are consistent with the routed specialist. "
                 "See routing receipt for confidence and per-step latency.").split()
        words = words[: max(1, min(max_new_tokens, len(words)))]
        t0 = time.perf_counter()
        per_token = []
        for i, word in enumerate(words):
            step0 = time.perf_counter()
            text = (" " if i else "") + word
            dt = (time.perf_counter() - step0) * 1000.0
            per_token.append(dt)
            yield {"event": "token", "index": i, "text": text,
                   "latency_ms": round(dt, 3),
                   "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
        total = (time.perf_counter() - t0) * 1000.0
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=0.0, decode_ms=total,
                                   tokens=len(words),
                                   tok_per_s=len(words) / max(total / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}

    def _real_stream(self, prompt: str, max_new_tokens: int,
                     temperature: float) -> Generator[Dict[str, Any], None, None]:
        import torch
        from transformers import TextIteratorStreamer

        enc = self._tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._model.device)
        t0 = time.perf_counter()
        pre0 = time.perf_counter()
        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids, streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            pad_token_id=self._tokenizer.pad_token_id,
        )
        thread = threading.Thread(
            target=lambda: self._model.generate(**gen_kwargs), daemon=True)
        thread.start()
        prefill_ms = (time.perf_counter() - pre0) * 1000.0
        per_token: list = []
        idx = 0
        # First yielded chunk includes prefill; per-token timing from 2nd chunk.
        prev = time.perf_counter()
        for chunk in streamer:
            now = time.perf_counter()
            dt = (now - prev) * 1000.0
            prev = now
            per_token.append(dt)
            yield {"event": "token", "index": idx, "text": chunk,
                   "latency_ms": round(dt, 3),
                   "elapsed_ms": round((now - t0) * 1000.0, 3)}
            idx += 1
        thread.join()
        total = (time.perf_counter() - t0) * 1000.0
        decode_ms = max(total - prefill_ms, 0.0)
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=prefill_ms, decode_ms=decode_ms,
                                   tokens=idx,
                                   tok_per_s=idx / max(decode_ms / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}
