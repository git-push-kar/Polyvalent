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
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

from router.latency import GenerationReceipt, TokenStat
from router.prompts import SYSTEM_BY_TASK, build_messages

logger = logging.getLogger(__name__)

_ANOMALY = "anomaly"
_REASONING = "reasoning"
_DEEPFAKE_LLM = "deepfake_llm"
_DEEPFAKE_VISION = "deepfake_vision"


def _assert_attached(caught, name: str) -> None:
    """Refuse silent base-model generation when PEFT could not map weights."""
    missing = [str(w.message)[:160] for w in caught
               if "missing adapter keys" in str(w.message)]
    if missing:
        raise RuntimeError(
            f"adapter {name!r} weights did not attach (key namespace "
            f"mismatch): {missing[0]}... Refusing silent base-model generation.")


class TagStripper:
    """Streaming XML-tag filter for non-forensic tasks.

    The LR-fused base leaks template closers (</question>, </answer>) even
    under a no-XML system prompt. Tags may split across token boundaries, so
    a trailing partial tag (`<` + letters, no `>` yet) is held back until it
    completes or is disproved. Plain-text uses of `<` (e.g. `a < b`) pass
    through: only `<letter...>` shapes are treated as tags.
    """

    _TAG_RE = None

    def __init__(self, max_carry: int = 32) -> None:
        import re
        if TagStripper._TAG_RE is None:
            TagStripper._TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
        self._carry = ""
        self._max_carry = max_carry

    def feed(self, text: str) -> str:
        data = self._carry + (text or "")
        self._carry = ""
        # Hold back a trailing partial tag.
        lt = data.rfind("<")
        if lt != -1 and ">" not in data[lt:]:
            tail = data[lt:]
            import re as _re
            if _re.match(r"</?[A-Za-z]*$", tail) and len(tail) <= self._max_carry:
                self._carry = tail
                data = data[:lt]
        return TagStripper._TAG_RE.sub("", data)

    def flush(self) -> str:
        tail, self._carry = self._carry, ""
        return TagStripper._TAG_RE.sub("", tail)


class TagSectioner:
    """Streaming section router for the forensic wrapper's tagged reasoning.

    The trained forensic LM emits
    `<fast>/<planning>/<reasoning>/<reflection>/<conclusion>/<answer>`
    sections. Everything except `<answer>` streams on the ``thinking`` channel
    (rendered collapsed, ChatGPT-style); `<answer>` streams on ``answer``.
    Unknown tags are dropped (name only) and never leak. Tags may split across
    token boundaries, so a trailing partial tag is held back until it completes
    or is disproved.

    `feed()` / `flush()` return ``[(channel, text), ...]`` with channel in
    ``{"thinking", "answer"}``. Accumulated ``answer`` / ``conclusion`` text is
    exposed for post-stream fallbacks.
    """

    _THINKING = ("fast", "planning", "reasoning", "reflection", "conclusion")

    def __init__(self, default: str = "answer", max_carry: int = 48) -> None:
        import re
        self._re = re.compile(r"</?([A-Za-z][\w.-]*)[^<>]*>")
        self._partial_re = re.compile(r"</?[A-Za-z][\w.-]*$")
        self._default = default
        self._section: Optional[str] = None
        self._carry = ""
        self._max_carry = max_carry
        self.answer_text: list = []
        self.conclusion_text: list = []
        self.thinking_text: list = []

    def _channel(self) -> str:
        if self._section in self._THINKING:
            return "thinking"
        if self._section == "answer":
            return "answer"
        return self._default

    def _push(self, text: str, out: list) -> None:
        if not text:
            return
        channel = self._channel()
        if self._section is None and not text.strip():
            return  # whitespace between sections — never the answer
        if channel == "thinking":
            self.thinking_text.append(text)
        if channel == "answer":
            self.answer_text.append(text)
        if self._section == "conclusion":
            self.conclusion_text.append(text)
        out.append((channel, text))

    def feed(self, text: str) -> list:
        data = self._carry + (text or "")
        self._carry = ""
        out: list = []
        while data:
            lt = data.find("<")
            if lt == -1:
                self._push(data, out)
                break
            if lt > 0:
                self._push(data[:lt], out)
                data = data[lt:]
            gt = data.find(">")
            if gt == -1:  # partial tag: hold if plausible, else emit "<"
                if len(data) <= self._max_carry and self._partial_re.match(data):
                    self._carry = data
                    break
                self._push("<", out)
                data = data[1:]
                continue
            tag = data[:gt + 1]
            m = self._re.match(tag)
            if m:
                name = m.group(1).lower()
                closing = tag.startswith("</")
                if name in self._THINKING or name == "answer":
                    self._section = None if closing else name
                # unknown tags are dropped (name + attrs), never leaked
            else:
                self._push("<", out)  # literal "<" (e.g. "a < b>")
                data = data[1:]
                continue
            data = data[gt + 1:]
        return out

    def flush(self) -> list:
        data, self._carry = self._carry, ""
        if not data:
            return []
        if self._partial_re.match(data):
            return []  # truncated tag at stream end
        out: list = []
        self._push(data, out)
        return out


class AdapterService:
    def __init__(self, base_dir: str,
                 anomaly_dir: str,
                 reasoning_dir: str,
                 deepfake_llm_dir: str,
                 deepfake_vision_dir: str,
                 custom_heads_path: Optional[str] = None) -> None:
        # All three adapters target the SAME pristine InternVL3-2B base.
        #   anomaly + reasoning -> one shared PeftModel (hot-swapped)
        #   deepfake            -> dedicated wrapper (own backbone + heads)
        self.base_dir = base_dir
        self.anomaly_dir = anomaly_dir
        self.reasoning_dir = reasoning_dir
        self.deepfake_llm_dir = deepfake_llm_dir
        self.deepfake_vision_dir = deepfake_vision_dir
        self.custom_heads_path = custom_heads_path
        self._model = None
        self._tokenizer = None
        self._active: Optional[str] = None
        self._deepfake = None          # (wrapper, tokenizer)
        self._deepfake_origin_dirs: Optional[Tuple[str, str]] = None

    # -- adapter selection --------------------------------------------
    def select(self, task: str) -> Dict[str, Any]:
        """Activate the adapter set for `task`. Returns selection receipt."""
        t0 = time.perf_counter()
        try:
            self._ensure_model()
        except Exception as exc:
            if os.getenv("FINAL_STUB_GENERATION", "0") == "1":
                logger.warning("model unavailable, stub selection: %s", exc)
                names = {"anomaly": [_ANOMALY],
                         "reasoning": [_REASONING],
                         "deepfake": [_DEEPFAKE_LLM, _DEEPFAKE_VISION,
                                      "custom_heads(wrapper)"],
                         "general": ["base"]}[task]
                return {"task": task, "status": 200, "adapters": names,
                        "mode": "stub",
                        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
            raise
        if task == "anomaly":
            self._set_active([_ANOMALY])
            self._active = _ANOMALY
            names = [_ANOMALY]
        elif task == "reasoning":
            self._set_active([_REASONING])
            self._active = _REASONING
            names = [_REASONING]
        elif task == "deepfake":
            # Dedicated wrapper owns deepfake end-to-end (own backbone +
            # submodule LoRA + classifier heads). Top model idles on base.
            self._set_active([])
            self._ensure_deepfake()
            self._active = "deepfake"
            names = ["deepfake_llm(wrapper)", "deepfake_vision(wrapper)",
                     "custom_heads(wrapper)"]
        elif task == "general":
            self._set_active([])  # plain base model
            self._active = None
            names = ["base"]
        else:
            raise ValueError(f"unknown task: {task}")
        return {"task": task, "status": 200, "adapters": names,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}

    def _set_active(self, names: list) -> None:
        """Activate adapters per tuner layer (peft 0.17-compatible).

        Layers ignore names missing from their own lora_A, so disjoint
        adapters (llm vs vision) compose safely; [] = plain base model.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer
        for module in self._model.modules():
            if isinstance(module, BaseTunerLayer) and hasattr(module, "lora_A"):
                module.set_adapter([n for n in names if n in module.lora_A])
        # Model-level tag is only used for config lookup (active_peft_config);
        # it must name a LOADED adapter even when every layer runs bare.
        self._model.active_adapter = (
            names[0] if names else next(iter(self._model.peft_config)))

    def _ensure_model(self) -> None:
        """Shared PeftModel: pristine base + anomaly + reasoning adapters.

        Only one adapter is active per request (hot-swapped in `select`);
        `general` runs the bare base. Deepfake lives in its own wrapper.
        """
        if self._model is not None:
            return
        import warnings
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
        # Required by InternVLChatModel.generate(): maps <IMG_CONTEXT> slots
        # to vision embeddings. Must be set on the BASE model before wrapping
        # (text-only calls pass pixel_values=None, so no slots are replaced).
        try:
            base.img_context_token_id = self._tokenizer.convert_tokens_to_ids(
                "<IMG_CONTEXT>")
        except Exception as exc:
            logger.warning("could not set img_context_token_id: %s", exc)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PeftModel.from_pretrained(
                base, self.anomaly_dir,
                adapter_name=_ANOMALY, is_trainable=False)
        _assert_attached(caught, _ANOMALY)
        # Reasoning is a second LoRA on the same base/target modules. PEFT
        # keeps both loaded; `_set_active` picks exactly one per request.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.load_adapter(self.reasoning_dir, adapter_name=_REASONING,
                               is_trainable=False)
        _assert_attached(caught, _REASONING)
        model.eval()
        self._model = model

    def _ensure_deepfake(self):
        """Dedicated DeepfakeReasoningModel wrapper (own backbone instance).

        Loads the ORIGINAL submodule-relative LoRA safetensors (the training
        harness wrapped backbone.language_model / .vision_model directly) +
        the 59 custom-head tensors. Any unmapped key raises — never silent.
        """
        if self._deepfake is not None:
            return self._deepfake
        import torch
        from deepfake.loader import load_trained_wrapper

        if not self.custom_heads_path:
            raise RuntimeError(
                "deepfake custom_heads path is required (FINAL_CUSTOM_HEADS)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._deepfake = load_trained_wrapper(
            base_dir=self.base_dir,
            llm_adapter_path=self.deepfake_llm_dir,
            vision_adapter_path=self.deepfake_vision_dir,
            heads_path=self.custom_heads_path,
            device=device,
        )
        self._deepfake_origin_dirs = ("llm_adapter", "vision_adapter")
        return self._deepfake

    # -- deepfake forensic analysis (wrapper path) -------------------------
    def analyze_image_stream(self, image_bytes: bytes, max_new_tokens: int = 512,
                             temperature: float = 0.0,
                             repetition_penalty: float = 1.05,
                             stub: Optional[bool] = None) -> Generator[Dict[str, Any], None, None]:
        """Full forensic pass: classifier verdict first, then streamed reasoning.

        The trained forensic LM emits <fast>/<planning>/<reasoning>/<reflection>/
        <conclusion>/<answer> XML tags. A streaming section router sends the
        reasoning sections on the "thinking" channel (rendered collapsed by the
        client) and <answer> on the "token" channel, so callers never see tags.

        Event shapes:
          {"event": "verdict", "verdict": "REAL"|"FAKE", "confidence": f,
           "hotspots": str, "spectral": str, "telemetry_ms": ms}
          {"event": "thinking", ...}  (fast/planning/reasoning/reflection/conclusion)
          {"event": "token", ...}     (the <answer> text, as it generates)
          {"event": "generation_receipt", ...}
        """
        import io as _io
        use_stub = (os.getenv("FINAL_STUB_GENERATION", "0") == "1"
                    if stub is None else stub)
        if use_stub:
            yield {"event": "verdict", "verdict": "REAL", "confidence": 0.5,
                   "hotspots": "stub", "spectral": "stub",
                   "telemetry_ms": 0.0, "mode": "stub"}
            yield from self._stub_stream("stub forensic reasoning", max_new_tokens)
            return
        import torch
        from PIL import Image as _Image
        from transformers import TextIteratorStreamer

        from deepfake.forensics import (build_telemetry_prompt,
                                        extract_forensic_telemetry,
                                        preprocess_image)
        wrapper, tokenizer = self._ensure_deepfake()
        device = next(wrapper.parameters()).device
        try:
            img = _Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            img.load()
        except Exception:
            yield {"event": "notice",
                   "text": "That file doesn't look like an image I can read. "
                           "Please attach a JPG or PNG face photo and ask again."}
            return
        pv, freq = preprocess_image(img, device=str(device))

        t_tele = time.perf_counter()
        with torch.no_grad():
            _vis = wrapper._get_vision_features(pv, return_attn_weights=True)
            cls_token, pooled, _patches, attn_w = _vis
            freq_emb = wrapper.freq_branch(freq.to(device))
            _dt = wrapper.cls_head.net[0].weight.dtype
            vis_cat = torch.cat([cls_token.to(_dt), pooled.to(_dt),
                                 freq_emb.to(_dt)], dim=-1)
            logits = wrapper.cls_head(vis_cat)
            probs = torch.softmax(logits.float(), dim=-1)
            pred = int(logits.argmax(-1).item())
            conf = float(probs[0, pred].item())
        telemetry = extract_forensic_telemetry(pred, conf, freq, attn_w)
        telemetry_ms = (time.perf_counter() - t_tele) * 1000.0
        yield {"event": "verdict", "verdict": telemetry["verdict"],
               "confidence": round(conf, 4),
               "hotspots": telemetry["regions"],
               "spectral": telemetry["spectral_desc"],
               "telemetry_ms": round(telemetry_ms, 1)}
        prompt_text = build_telemetry_prompt(telemetry)

        # Reasoning generation with token streaming (mirrors predict()).
        vis_tokens = wrapper._get_native_visual_tokens(pv)
        inputs_embeds = vis_tokens.to(torch.bfloat16)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        prompt_embeds = wrapper.backbone.language_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([inputs_embeds,
                                   prompt_embeds.to(inputs_embeds.dtype)], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                    device=device)
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
            stop_ids = tokenizer("</answer>", add_special_tokens=False).input_ids

            class _StopOnAnswer(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    tail = input_ids[0, -len(stop_ids):].tolist()
                    return tail == stop_ids

            stopping = StoppingCriteriaList([_StopOnAnswer()])
        except Exception:
            stopping = None
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        gen_kwargs: Dict[str, Any] = dict(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            streamer=streamer, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        t0 = time.perf_counter()
        pre0 = time.perf_counter()
        thread = threading.Thread(
            target=lambda: self._generate_guarded_lm(wrapper, gen_kwargs, streamer),
            daemon=True)
        thread.start()
        prefill_ms = (time.perf_counter() - pre0) * 1000.0
        per_token: list = []
        idx = 0
        sectioner = TagSectioner(default="answer")
        prev = time.perf_counter()
        for chunk in streamer:
            now = time.perf_counter()
            dt = (now - prev) * 1000.0
            prev = now
            per_token.append(dt)
            idx += 1
            for channel, text in sectioner.feed(chunk):
                yield {"event": "thinking" if channel == "thinking" else "token",
                       "index": idx, "text": text,
                       "latency_ms": round(dt, 3),
                       "elapsed_ms": round((now - t0) * 1000.0, 3)}
        thread.join()
        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        for channel, text in sectioner.flush():  # release any partial-tag carry
            yield {"event": "thinking" if channel == "thinking" else "token",
                   "index": idx, "text": text, "latency_ms": 0.0,
                   "elapsed_ms": elapsed}
        # If the run stopped before <answer> (budget/stop), surface the
        # conclusion as the answer so the client never shows an empty reply.
        if not sectioner.answer_text:
            fallback = "".join(sectioner.conclusion_text).strip()
            if fallback:
                yield {"event": "token", "index": idx, "text": fallback,
                       "latency_ms": 0.0, "elapsed_ms": elapsed}
        if gen_kwargs.get("_worker_error"):
            yield {"event": "notice",
                   "text": "Reasoning hit an error partway; verdict above stands. "
                           f"Detail: {gen_kwargs['_worker_error'][:200]}"}
        total = (time.perf_counter() - t0) * 1000.0
        decode_ms = max(total - prefill_ms, 0.0)
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=prefill_ms, decode_ms=decode_ms,
                                   tokens=idx,
                                   tok_per_s=idx / max(decode_ms / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}

    def _generate_guarded_lm(self, wrapper, gen_kwargs: Dict[str, Any],
                             streamer: Any) -> None:
        try:
            wrapper.backbone.language_model.generate(**gen_kwargs)
        except Exception as exc:
            logger.warning("deepfake reasoning failed, ending stream: %s\n%s",
                           exc, traceback.format_exc())
            gen_kwargs["_worker_error"] = str(exc)
        finally:
            try:
                streamer.end()
            except Exception:
                pass
        self._heads_loaded = True

    # -- streaming generation ------------------------------------------
    def stream(self, prompt: str, max_new_tokens: int = 256,
               temperature: float = 0.0, repetition_penalty: float = 1.0,
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
        yield from self._real_stream(prompt, max_new_tokens, temperature,
                                     repetition_penalty)

    def _stub_required(self) -> bool:
        try:
            self._ensure_model()
            return False
        except Exception as exc:  # weights/GPU absent: stub keeps shape testable
            logger.warning("model unavailable, stub stream: %s", exc)
            return True

    def format_prompt(self, task: str, history: list, message: str) -> str:
        """Training-faithful prompt via the fused tokenizer's own template.

        Bare user text lets the LR-fused base fall back to its logic-rule
        prior (rule traces, token loops). The adapters were trained under
        <|im_start|>system framing — reproduce it here.
        """
        messages = build_messages(task, history or [], message)
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:  # pragma: no cover - very old tokenizers
            logger.warning("chat template failed, manual framing: %s", exc)
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    def eos_ids(self) -> list:
        ids = {self._tokenizer.eos_token_id}
        try:
            extra = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(extra, int) and extra > 0:
                ids.add(extra)
        except Exception:
            pass
        return [i for i in ids if i]

    def _generate_guarded(self, gen_kwargs: Dict[str, Any],
                            streamer: Any) -> None:
        """Run generate() off-thread; never leave the SSE iterator hanging.

        On worker error the exception is logged and the streamer is ended so
        the consumer sees ... + generation_receipt(partial) + done instead of
        blocking forever.
        """
        try:
            self._model.generate(**gen_kwargs)
        except Exception as exc:
            logger.warning("generation failed, ending stream: %s\n%s",
                           exc, traceback.format_exc())
            gen_kwargs["_worker_error"] = str(exc)
        finally:
            try:
                streamer.end()
            except Exception:
                pass

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
                     temperature: float,
                     repetition_penalty: float = 1.0) -> Generator[Dict[str, Any], None, None]:
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
            repetition_penalty=repetition_penalty,
            eos_token_id=self.eos_ids(),
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        thread = threading.Thread(
            target=lambda: self._generate_guarded(gen_kwargs, streamer),
            daemon=True)
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
        if gen_kwargs.get("_worker_error"):
            yield {"event": "notice",
                   "text": "Generation hit an error partway; showing what arrived. "
                           f"Detail: {gen_kwargs['_worker_error'][:200]}"}
        total = (time.perf_counter() - t0) * 1000.0
        decode_ms = max(total - prefill_ms, 0.0)
        yield {"event": "generation_receipt",
               **GenerationReceipt(prefill_ms=prefill_ms, decode_ms=decode_ms,
                                   tokens=idx,
                                   tok_per_s=idx / max(decode_ms / 1000.0, 1e-9),
                                   per_token_ms=per_token).to_dict()}
