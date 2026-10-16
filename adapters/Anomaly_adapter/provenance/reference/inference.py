"""InternVL3-2B / InternVL2-2B + tep_rca inference: automatic reports and follow-up answers.

This module is the LLM-facing half of the pipeline. It receives structured
anomaly events (never raw sensor series) and returns structured reports or
conversational answers grounded in the event evidence and prior context.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from llm.adapter_loader import load_tep_adapter
from llm.dataset import (
    ANSWER_KEYS,
    EOS,
    SYSTEM_PROMPT,
    build_chat_prompt,
    format_evidence_question,
)
from utils import load_config

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = (
    "You are an industrial process anomaly diagnosis assistant. A new anomaly "
    "has automatically been detected. Analyze the provided evidence and "
    "generate a complete initial report WITHOUT waiting for the user to ask a "
    "question."
)


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering on a batch x vocab probability tensor."""
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    cumsum = sorted_probs.cumsum(dim=-1)
    keep = cumsum - sorted_probs <= top_p
    sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_json_object(text: str) -> Optional[Dict]:
    """Extract the first balanced JSON object from a model response."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode failed: %s", exc)
                    return None

    # Fallback: attempt to recover if JSON was truncated near the end
    if depth > 0:
        candidate = text[start:].strip()
        if in_string:
            candidate += '"'
        for _ in range(depth):
            candidate += "}"
        try:
            repaired = json.loads(candidate)
            if isinstance(repaired, dict):
                logger.info("Successfully recovered truncated JSON object from model output.")
                return repaired
        except Exception:
            pass

    return None


def sanitize_answer(answer: Dict, evidence: Optional[Dict] = None) -> Dict:
    """Fill in missing answer keys so reports always match the schema."""
    if not isinstance(answer, dict):
        answer = {}
    if evidence is not None and not isinstance(evidence, dict):
        evidence = {}
    for key in ANSWER_KEYS:
        answer.setdefault(key, "" if key in ("summary", "root_cause",
                                             "affected_subsystem", "reasoning",
                                             "reasoning_steps",
                                             "recommended_action", "uncertainty",
                                             "evidence") else ("unknown" if key == "severity" else 0.0))
    if not isinstance(answer["evidence"], list):
        answer["evidence"] = [str(answer["evidence"])]
    if answer.get("confidence") is None or not (0.0 <= float(answer["confidence"]) <= 1.0):
        answer["confidence"] = 0.0
    if answer["severity"] not in ("low", "medium", "high", "critical"):
        answer["severity"] = "unknown"

    # Deterministic severity guardrail: If the model failed to output valid severity or if evidence provides deterministic rating
    if evidence and answer["severity"] in ("unknown", ""):
        from evidence.process_limits import compute_deterministic_severity
        answer["severity"] = evidence.get("severity") or compute_deterministic_severity(evidence)
    return answer


class RCAInference:
    """High-level LLM wrapper around (InternVL2-2B + tep_rca)."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.config = config or load_config()
        self.model = model
        self.tokenizer = tokenizer
        self._gen_config = self.config["llm"]["generation"]

    # ------------------------------------------------------------------
    @classmethod
    def from_adapter(
        cls,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> "RCAInference":
        from transformers import AutoTokenizer

        config = config or load_config()
        llm = config["llm"]
        base = base_model or llm["base_model"]
        adapter = adapter_path or llm["adapter_dir"]
        adapter_name = llm.get("adapter_name", "tep_rca")
        model = load_tep_adapter(
            base,
            adapter,
            trust_remote_code=bool(llm.get("trust_remote_code", True)),
            use_flash_attn=bool(llm.get("use_flash_attn", False)),
            adapter_name=adapter_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base, trust_remote_code=True, use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model=model, tokenizer=tokenizer, config=config)

    def _generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Greedy decode via repeated forward passes.

        The base InternVL2 remote code predates the KV-cache APIs used by the
        installed transformers, so ``model.generate`` is unreliable here. We
        instead run the plain forward pass (which the LoRA training path uses
        successfully) and append argmax tokens until EOS or the token budget.
        """
        eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS)
        max_new = max_new_tokens or int(self._gen_config.get("max_new_tokens", 384))
        if do_sample is None:
            do_sample = bool(self._gen_config.get("do_sample", False))
        temp = float(self._gen_config.get("temperature", 0.7) if temperature is None else temperature)
        top_p = float(self._gen_config.get("top_p", 0.9) if top_p is None else top_p)
        top_k = int(self._gen_config.get("top_k", 50) if top_k is None else top_k)
        rep_penalty = float(self._gen_config.get("repetition_penalty", 1.0) if repetition_penalty is None else repetition_penalty)
        pad = self.tokenizer.pad_token_id or eos_token_id

        # Collect all valid stop token IDs (EOS, <|im_end|>, tokenizer eos, etc.)
        stop_token_ids = {eos_token_id, self.tokenizer.eos_token_id}
        for token_str in (EOS, "<|im_end|>", "</s>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and isinstance(tid, int) and tid > 0:
                stop_token_ids.add(tid)
        stop_token_ids.discard(None)

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        # Text-only InternVL2 forward needs the dummy image placeholders for prefill.
        dummy = torch.zeros(
            1, 3, int(self.config["llm"].get("dummy_image_size", 448)),
            int(self.config["llm"].get("dummy_image_size", 448)),
            dtype=torch.bfloat16, device=self.model.device,
        )
        image_flags = torch.zeros((input_ids.shape[0],), dtype=torch.long, device=self.model.device)

        generated: List[int] = []
        past_key_values = None
        use_cache = True
        next_id = None

        with torch.no_grad():
            for step in range(max_new):
                if step == 0:
                    # Step 0 (Prefill): evaluate prompt and initialize KV cache
                    try:
                        outputs = self.model(
                            pixel_values=dummy,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            use_cache=True,
                        )
                        past_key_values = getattr(outputs, "past_key_values", None)
                        logits = outputs.logits[:, -1, :]
                        if past_key_values is None:
                            use_cache = False
                    except Exception as exc:
                        logger.warning("KV caching prefill not supported (%s); using direct decode.", exc)
                        use_cache = False
                        logits = self.model(
                            pixel_values=dummy,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            use_cache=False,
                        ).logits[:, -1, :]
                else:
                    # Autoregressive generation steps
                    if use_cache and past_key_values is not None:
                        try:
                            # Fast path: pass only the 1 newly generated token with KV cache (O(1))
                            outputs = self.model(
                                pixel_values=None,
                                input_ids=next_id,
                                attention_mask=attention_mask,
                                image_flags=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            past_key_values = outputs.past_key_values
                            logits = outputs.logits[:, -1, :]
                        except Exception:
                            # Fallback if pixel_values=None is rejected by custom model architecture
                            try:
                                outputs = self.model(
                                    pixel_values=dummy,
                                    input_ids=next_id,
                                    attention_mask=attention_mask,
                                    image_flags=image_flags,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                )
                                past_key_values = outputs.past_key_values
                                logits = outputs.logits[:, -1, :]
                            except Exception as exc:
                                logger.warning("KV cache step failed (%s); reverting to full sequence.", exc)
                                use_cache = False
                                logits = self.model(
                                    pixel_values=dummy,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    image_flags=image_flags,
                                    use_cache=False,
                                ).logits[:, -1, :]
                    else:
                        logits = self.model(
                            pixel_values=dummy,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_flags=image_flags,
                            use_cache=False,
                        ).logits[:, -1, :]

                if rep_penalty != 1.0 and generated:
                    for g in set(generated):
                        val = logits[:, g]
                        logits[:, g] = torch.where(val < 0, val * rep_penalty, val / rep_penalty)

                if do_sample:
                    logits = logits / temp
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    if top_p < 1.0:
                        probs = _apply_top_p(probs, top_p)
                    if top_k > 0:
                        top_probs, _ = torch.topk(probs, min(top_k, probs.shape[-1]))
                        probs[probs < top_probs[:, -1:]] = 0.0
                        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    if top_k > 0:
                        top_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                        logits[logits < top_vals[:, -1:]] = -float("inf")
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                next_id = next_id.to(self.model.device)
                curr_token = int(next_id.item())
                if curr_token in stop_token_ids:
                    break

                generated.append(curr_token)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_id)], dim=-1
                )

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        for stop_str in (EOS, "<|im_end|>", "</s>", "<|endoftext|>"):
            text = text.replace(stop_str, "")
        return text.strip()

    # ------------------------------------------------------------------
    def generate_report(self, evidence: Dict) -> Tuple[Dict, str]:
        """Automatic initial report for an anomaly event.

        Returns:
            (report_dict, raw_text).
        """
        question = format_evidence_question(evidence)
        prompt = build_chat_prompt(REPORT_SYSTEM_PROMPT + " " + SYSTEM_PROMPT, [("user", question)])
        raw = self._generate(prompt, max_new_tokens=int(self._gen_config.get("max_new_tokens", 768)))
        parsed = extract_json_object(raw)
        report = sanitize_answer(parsed, evidence=evidence) if parsed else {
            "reasoning_steps": "Fallback deterministic diagnosis based on sensor limits.",
            "summary": raw,
            "root_cause": "unable_to_parse",
            "affected_subsystem": evidence.get("candidate_subsystem", "unknown"),
            "evidence": [],
            "reasoning": raw,
            "severity": evidence.get("severity", "unknown"),
            "confidence": 0.0,
            "recommended_action": "Review the raw evidence and re-run analysis.",
            "uncertainty": "The model response could not be parsed as JSON.",
        }
        return report, raw

    def answer_followup(
        self,
        event: Dict,
        previous_report: Dict,
        history: List[Dict],
        question: str,
    ) -> str:
        """Answer a user question about a specific detected anomaly.

        Every question is processed directly by the LLM using comprehensive
        incident telemetry context and the fine-tuned domain adapter. No
        canned keyword templates are forced.
        """
        chat_system_prompt = (
            "You are an industrial process assistant and control room diagnostic expert for the "
            "Tennessee Eastman Process (TEP). An operator is asking questions about a specific "
            "detected anomaly event.\n\n"
            "You are provided with comprehensive incident telemetry, including model-level detector scores, "
            "individual sensor values and ideal baselines, the 52-sensor plant baseline, and diagnostic findings.\n\n"
            "GUIDELINES FOR ANSWERING:\n"
            "1. Answer the operator's question directly, accurately, and concisely based strictly on the incident data provided.\n"
            "2. When asked for 'sensor values', 'measured values', or 'ideal/normal values':\n"
            "   - Extract and state the exact numbers directly from the 'Sensor Telemetry' section (Observed Value vs Ideal/Normal Baseline, with deviation %).\n"
            "   - NEVER say 'as shown in the table below' or give vague non-answers; list the actual numbers.\n"
            "3. When asked for 'scores calling this an anomaly' or detector scores:\n"
            "   - Report the overall anomaly score, LSTM Autoencoder error vs threshold, and CVA Q-statistic vs threshold.\n"
            "4. When asked 'what is the evidence which forces this to be an anomaly':\n"
            "   - Clearly cite the active detector threshold exceedances (e.g. CVA Q-statistic and/or LSTM reconstruction error alarms) and list the primary deviating sensors that triggered the event.\n"
            "5. If asked which sensors are used in this system: explain that the plant monitors 52 continuous process variables (41 measurement variables XMEAS 1 to 41 and 11 manipulated variables XMV 1 to 11), and state which sensors are currently deviating in this incident.\n"
            "6. If asked for a fix or mitigation: outline specific engineering corrective actions, valves or loops to inspect, and safety precautions based on the suspected root cause and affected subsystem.\n"
            "7. If asked for severity: state the exact assessed level (low, medium, high, or critical) and the safety rationale.\n"
            "8. Answer fresh and directly in professional engineering language. Do NOT emit special tokens (such as </s>) or repeat boilerplate sentences from previous turns.\n\n"
            "VOICE AND STYLE (write like a top-tier assistant: direct, structured, conversational-professional):\n"
            "1. Lead with the direct answer in the first 1-2 sentences. Never open with throat-clearing "
            "('The evidence is not sufficient...', 'It is important to note...').\n"
            "2. Commit to the single best-candidate cause with an explicit confidence "
            "(high/medium/low) drawn from the telemetry; put residual doubt in ONE closing "
            "uncertainty line plus the on-site verification step. Reserve 'unclassified' for "
            "cases where no detector alarmed and no sensor deviated — never use it when a "
            "detector threshold was exceeded.\n"
            "3. Structure longer answers with short markdown sections and bullets; keep paragraphs "
            "to 2-3 sentences. Numbers go in markdown tables (one row per detector/sensor), "
            "never in bare 'Name (TAG, ...)' lists — every sensor row carries its numbers.\n"
            "4. Explain the 'why' mechanistically in one or two sentences (initiating variable -> "
            "propagation), not as a recited template. Vary your phrasing; do not reuse stock sentences "
            "like 'The key point is...'.\n"
            "5. Close action answers with the concrete next physical step and its safety precaution."
        )

        if not isinstance(event, dict):
            event = {}
        if not isinstance(previous_report, dict):
            previous_report = {}

        context = self._render_event_context(event, previous_report)
        ev_id = event.get("event_id", "UNKNOWN")
        rc = previous_report.get("root_cause", "Unknown")
        sev = str(previous_report.get("severity", "Unknown")).upper()

        messages: List[Tuple[str, str]] = [
            ("user", f"Here is the telemetry, detector scores, sensor metrics, and plant context for anomaly {ev_id}:\n\n{context}"),
            ("assistant", f"I have reviewed the diagnostic data for incident {ev_id}. Suspected cause: {rc}, Severity: {sev}. How can I assist you with this event?"),
        ]

        # Bound history to last 2 turns (1 exchange) to keep context sharp, prevent token pollution,
        # and eliminate repetition loops from previous turns.
        recent_history = history[-2:] if isinstance(history, list) else []
        for msg in recent_history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                role = msg[0]
                content = msg[1]
            else:
                continue
            if role in ("user", "assistant") and content:
                messages.append((str(role), str(content)))

        messages.append(("user", question))
        prompt = build_chat_prompt(chat_system_prompt, messages)

        # Retain the fine-tuned TEP adapter for full chemical process domain knowledge.
        # Use greedy decoding (do_sample=False) for grounded, deterministic engineering answers.
        # Budget raised (Track 2): open-ended answers need room for exact
        # numbers; deterministic intents already bypass the LLM entirely.
        return self._generate(
            prompt,
            max_new_tokens=max(384, min(int(self._gen_config.get("max_new_tokens", 512)), 512)),
            do_sample=False,
            repetition_penalty=1.12,
        )

    @classmethod
    def _dispatch_operator_intent(
        cls,
        event: Dict,
        report: Dict,
        question: str,
    ) -> Optional[str]:
        """Detect and immediately answer common operational control-room questions."""
        hit = cls._dispatch_with_intent(event, report, question)
        return hit[1] if hit is not None else None

    @classmethod
    def _guard_llm_answer(
        cls,
        event: Dict,
        report: Dict,
        question: str,
        llm_answer: str,
    ) -> Optional[Tuple[str, str]]:
        """Numeric guardrail for LLM answers.

        If the operator asked for scores / evidence / sensor values but the
        LLM answer contains no digits (or uses vague boilerplate instead of
        numbers), fall back to the deterministic intent answer.
        Returns (intent, answer) on guardrail trip, else None.
        """
        if not isinstance(llm_answer, str):
            return None
        q = (question or "").lower().strip()
        needs_numbers = any(w in q for w in (
            "score", "scores", "metric", "metrics", "threshold",
            "reconstruction error", "q-stat", "q stat",
            "evidence", "why is this an anomaly", "why was this flagged",
            "sensor value", "sensor values", "ideal value", "normal value",
            "baseline value", "measured value", "observed value",
        ))
        vague = any(p in llm_answer.lower() for p in (
            "as shown in the table below",
            "as shown below",
        ))
        has_digit = any(ch.isdigit() for ch in llm_answer)
        if needs_numbers and (not has_digit or vague):
            hit = cls._dispatch_with_intent(event, report, question)
            if hit is not None:
                return hit
            # Question demanded numbers but no intent matched: force evidence view.
            forced = cls._dispatch_with_intent(event, report, "provide the evidence for calling this an anomaly")
            if forced is not None:
                return forced
            return None
        # Foreign-sensor check: invented sensor-like tokens (not in evidence
        # display/tag names or subsystem) fail the answer even when digits
        # are present (e.g. hallucinated "Initiation_Xmeas_.." listings).
        if needs_numbers and isinstance(event, dict):
            def _norm(text: str) -> str:
                return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip().replace(" ", "_")
            evidence = event.get("evidence", {}) if isinstance(event.get("evidence"), dict) else {}
            sensors = evidence.get("top_anomalous_sensors", []) or []
            allowed = set()
            for s in sensors:
                if not isinstance(s, dict):
                    continue
                for key in ("display_name", "name"):
                    val = s.get(key, "")
                    if val:
                        allowed.add(_norm(str(val)))
            allowed.add(_norm(str(evidence.get("candidate_subsystem", ""))))
            sensor_re = re.compile(r"(?:^|[\s\W])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|X(?:MEAS|MV|MAX|MIN|PV|CV)[0-9]+)(?:$|[\s\W])")
            mentioned = set(t.lower() for t in sensor_re.findall(llm_answer))
            foreign = {m for m in mentioned
                       if not any(a.startswith(m) or m.startswith(a) for a in allowed if a)}
            if foreign:
                logger.warning("LLM answer named foreign sensors %s; using deterministic answer.",
                               sorted(foreign))
                hit = cls._dispatch_with_intent(event, report, question)
                if hit is not None:
                    return hit
                forced = cls._dispatch_with_intent(event, report, "provide the evidence for calling this an anomaly")
                if forced is not None:
                    return forced
        return None

    #: Intents whose answers must carry markdown tables with exact numbers.
    NUMERIC_FORMAT_INTENTS = frozenset({
        "sensor_values", "evidence_why", "sensor_scores",
        "detector_scores", "detector_severity",
    })

    #: Intents whose answers must name at least one deviating sensor.
    SENSOR_NAME_INTENTS = frozenset({
        "overview", "overview_fallback", "sensors_inspect",
        "system_sensors", "severity",
    })

    @classmethod
    def _evidence_markdown_tables(cls, event: Dict, report: Dict) -> str:
        """Build detector + sensor markdown tables straight from evidence.

        Used to repair LLM answers that list sensors/scores without numbers.
        """
        if not isinstance(event, dict):
            event = {}
        if not isinstance(report, dict):
            report = {}
        evidence = event.get("evidence", {}) if isinstance(event.get("evidence"), dict) else {}
        det_ev = evidence.get("detector_evidence", {}) if isinstance(evidence.get("detector_evidence"), dict) else {}
        score = float(event.get("max_anomaly_score", evidence.get("anomaly_score", 0.0)))
        ev_id = event.get("event_id", "UNKNOWN")

        lstm = det_ev.get("lstm_ae") if isinstance(det_ev.get("lstm_ae"), dict) else {}
        l_sc = float(lstm.get("score", 0.0))
        l_th = float(lstm.get("threshold", 0.0))
        l_state = "ALARMED" if (lstm.get("alarmed") or lstm.get("is_anomalous")) else "Normal (below limit)"
        dyn = det_ev.get("dynamic") if isinstance(det_ev.get("dynamic"), dict) else {}
        cva = dyn.get("cva") if isinstance(dyn.get("cva"), dict) else {}
        q_sc = float(cva.get("score", 0.0))
        q_th = float(cva.get("threshold", 0.0))
        q_state = "ALARMED" if (cva.get("alarmed") or cva.get("is_anomalous")) else "Normal (below limit)"

        lines = [
            f"**Telemetry tables for {ev_id} (from evidence, overall score `{score:.3f}`):**",
            "",
            "| Detector | Score | Threshold | Status |",
            "|---|---|---|---|",
            f"| LSTM reconstruction error | {l_sc:.4f} | {l_th:.4f} | {l_state} |",
            f"| CVA Q-statistic | {q_sc:.1f} | {q_th:.1f} | {q_state} |",
            "",
            "| Sensor | Observed | Baseline | Dev% | z | Trend |",
            "|---|---|---|---|---|---|",
        ]
        top = evidence.get("top_anomalous_sensors", []) or []
        if isinstance(top, list) and top:
            for s in top[:5]:
                if not isinstance(s, dict):
                    continue
                name = s.get("display_name", s.get("name", "Unknown"))
                cur_v = s.get("current_value")
                base_v = s.get("baseline_value")
                obs = f"{float(cur_v):.4f}" if cur_v is not None else "n/a"
                base = f"{float(base_v):.4f}" if base_v is not None else "n/a"
                dev = float(s.get("deviation_percent", 0.0))
                z = float(s.get("z_score", s.get("zScore", 0.0)))
                lines.append(f"| {name} | {obs} | {base} | {dev:+.2f}% | {z:+.2f} | {s.get('trend', 'unknown')} |")
        else:
            lines.append("| (no sensor excursions) | n/a | n/a | n/a | n/a | n/a |")
        return "\n".join(lines)

    @classmethod
    def format_followup_answer(
        cls,
        question: str,
        llm_answer: str,
        event: Dict,
        report: Dict,
    ) -> Tuple[str, Optional[str], str]:
        """Validate and repair an LLM follow-up answer's formatting.

        - Numeric intents without a markdown table get the evidence tables
          appended (numbers come from evidence, never invented).
        - Sensor intents naming none of the top-3 sensors get a
          "**Primary sensors:** ..." line appended.
        Returns (answer, intent, status) with status in
        {"ok", "repaired", "unrepairable"}.
        """
        if not isinstance(llm_answer, str) or not llm_answer.strip():
            return llm_answer, None, "unrepairable"
        hit = cls._dispatch_with_intent(event, report, question or "")
        intent = hit[0] if hit is not None else None
        if intent is None:
            return llm_answer, None, "ok"

        if not isinstance(event, dict):
            event = {}
        evidence = event.get("evidence", {}) if isinstance(event.get("evidence"), dict) else {}
        top = evidence.get("top_anomalous_sensors", []) or []
        top_names = [s.get("display_name", s.get("name", ""))
                     for s in top[:3] if isinstance(s, dict)]

        table_lines = sum(1 for ln in llm_answer.splitlines()
                          if ln.strip().startswith("|"))
        repaired = False
        answer = llm_answer
        if intent in cls.NUMERIC_FORMAT_INTENTS and table_lines < 3:
            det_ev = evidence.get("detector_evidence", {})
            if det_ev or top:
                answer = answer.rstrip() + "\n\n" + cls._evidence_markdown_tables(event, report)
                repaired = True
            else:
                return answer, intent, "unrepairable"
        if intent in cls.SENSOR_NAME_INTENTS and top_names:
            low = answer.lower()
            if not any(nm and nm.lower() in low for nm in top_names):
                answer = answer.rstrip() + "\n\n**Primary sensors:** " + ", ".join(top_names) + "."
                repaired = True
        return answer, intent, ("repaired" if repaired else "ok")

    @classmethod
    def _dispatch_with_intent(
        cls,
        event: Dict,
        report: Dict,
        question: str,
    ) -> Optional[Tuple[str, str]]:
        """Same as _dispatch_operator_intent but also returns the intent name."""
        # Intent labels mirror the sections below; used for route logging.
        if not isinstance(event, dict):
            event = {}
        if not isinstance(report, dict):
            report = {}
        q = question.lower().strip()
        ev_id = event.get("event_id", "UNKNOWN")
        evidence = event.get("evidence", {}) if isinstance(event.get("evidence"), dict) else {}
        det_ev = evidence.get("detector_evidence", {}) if isinstance(evidence.get("detector_evidence"), dict) else {}
        sev = str(report.get("severity") or evidence.get("severity", "unknown")).upper()
        score = float(event.get("max_anomaly_score", evidence.get("anomaly_score", 0.0)))
        root_cause = report.get("root_cause", "Process disturbance")
        subsystem = report.get("affected_subsystem") or evidence.get("candidate_subsystem", "unknown")
        action = report.get("recommended_action", "Inspect suspect equipment on site.")
        uncertainty = report.get("uncertainty", "On-site verification required to confirm causation.")

        # Disambiguate intents
        has_system_sensors = any(w in q for w in ("which sensors are used", "sensors used in this system", "what sensors are in", "sensors in the system", "system sensors", "all sensors", "sensors used"))
        has_sensor_values = not has_system_sensors and any(w in q for w in ("sensor value", "sensor values", "ideal value", "ideal values", "normal value", "normal values", "baseline value", "baseline values", "measured value", "measured values", "observed value", "observed values"))
        has_evidence_why = any(w in q for w in ("evidence", "forces this to be an anomaly", "why is this an anomaly", "why was this flagged", "alarm trigger", "why this is an anomaly", "why anomaly"))
        has_sensor_scores = not has_system_sensors and not has_sensor_values and (("sensor" in q and "score" in q) or "sensor score" in q)
        has_detector_scores = not has_sensor_scores and not has_system_sensors and not has_sensor_values and any(w in q for w in ("score", "scores", "metric", "metrics", "reconstruction error", "q-stat", "q stat", "q-score", "threshold"))
        has_severity = any(w in q for w in ("severity", "how serious", "how dangerous", "critical", "alarm level", "risk level", "danger"))
        has_sensors = not has_system_sensors and not has_sensor_scores and not has_sensor_values and any(w in q for w in ("sensor", "sensors", "inspect", "deviat", "which sensor"))
        has_action = any(w in q for w in ("what should i do", "what to do", "action", "recommend", "how to fix", "provide a fix", "fix", "mitigat", "next step", "procedure"))
        has_overview = any(w in q for w in ("which anomaly", "what is this anomaly", "what anomaly", "which fault", "what fault", "fault number", "fault type", "what happened", "what is this", "overview", "root cause", "cause", "explain anomaly", "explain this anomaly", "diagnose", "summary"))
        # Generic why/explain fallback intent (checked last so specific
        # metric/evidence/severity intents keep priority).
        has_generic_why = any(w in q for w in ("why ", "why?", "explain", "diagnos"))

        lstm = det_ev.get("lstm_ae") if isinstance(det_ev.get("lstm_ae"), dict) else {}
        l_sc = float(lstm.get("score", 0.0))
        l_th = float(lstm.get("threshold", 0.0))
        l_ratio = l_sc / max(l_th, 1e-9)
        is_l_alarm = bool(lstm.get("alarmed") or lstm.get("is_anomalous") or (l_sc >= l_th and l_th > 0))
        l_alarm = f"ALARMED (Active — {l_ratio:.2f}x threshold)" if is_l_alarm else "Normal (Below limit)"

        dyn = det_ev.get("dynamic") if isinstance(det_ev.get("dynamic"), dict) else {}
        cva = dyn.get("cva") if isinstance(dyn.get("cva"), dict) else {}
        q_sc = float(cva.get("score", 0.0))
        q_th = float(cva.get("threshold", 0.0))
        q_ratio = q_sc / max(q_th, 1e-9)
        is_q_alarm = bool(cva.get("alarmed") or cva.get("is_anomalous") or (q_sc >= q_th and q_th > 0))
        q_alarm = f"ALARMED (Active — {q_ratio:.2f}x threshold)" if is_q_alarm else "Normal (Below limit)"
        dom_lag = cva.get("dominant_lag", "N/A")
        slow_drift = float(cva.get("slow_drift_ratio", 0.0))

        # Intent: Sensor values and ideal normal baselines
        if has_sensor_values:
            top_sensors = evidence.get("top_anomalous_sensors", [])
            lines = [f"### Sensor Measured Values vs Ideal Normal Baseline for Incident {ev_id}", ""]
            if top_sensors:
                for s in top_sensors:
                    name = s.get("display_name", s.get("name", "Unknown"))
                    tag = s.get("name", "")
                    tag_str = f" (`{tag}`)" if tag and tag != name else ""
                    cur_v = s.get("current_value")
                    base_v = s.get("baseline_value")
                    dev = s.get("deviation_percent", 0.0)
                    z = s.get("z_score", s.get("zScore", 0.0))
                    trend = s.get("trend", "unknown")
                    contrib = s.get("contribution", 0.0)
                    lim = s.get("limit_context", "")
                    lim_text = f" -> **[Safety Envelope: {lim}]**" if lim else ""

                    if cur_v is not None and base_v is not None:
                        lines.append(
                            f"- **{name}**{tag_str}: Observed Value = `{float(cur_v):.4f}`, "
                            f"Ideal Normal Baseline = `{float(base_v):.4f}`, "
                            f"Deviation = `{dev:+.2f}%`, z-score = `{z:+.2f}`, Trend: `{trend}`{lim_text}"
                        )
                    else:
                        lines.append(
                            f"- **{name}**{tag_str}: Deviation = `{dev:+.2f}%`, "
                            f"z-score = `{z:+.2f}`, Trend: `{trend}`, Contribution = `{contrib:.2f}`{lim_text}"
                        )
            else:
                lines.append("No univariate sensor excursions; multi-sensor correlation dynamics triggered the anomaly.")
            return ("sensor_values", "\n".join(lines))

        # Intent: Why is this an anomaly / what evidence forces it
        if has_evidence_why:
            top_sensors = evidence.get("top_anomalous_sensors", [])
            lines = [
                f"### Evidence Triggering Anomaly {ev_id}",
                f"- **Overall Anomaly Score:** `{score:.3f}`",
                "",
                "**1. Detector Threshold Alarms:**",
                f"- **LSTM Autoencoder:** Reconstruction error = `{l_sc:.4f}` (Threshold: `{l_th:.4f}`, Margin: `{l_ratio:.2f}x` — {l_alarm})",
                f"- **CVA Dynamics Q-Statistic:** Score = `{q_sc:.1f}` (Threshold: `{q_th:.1f}`, Margin: `{q_ratio:.2f}x` — {q_alarm})",
                f"- **Dynamic Lag Profile:** Dominant lag `{dom_lag}`, Slow drift ratio `{slow_drift:.2f}`",
                "",
                "**2. Process Telemetry Excursions:**",
            ]
            if top_sensors:
                for s in top_sensors[:4]:
                    name = s.get("display_name", s.get("name", "Sensor"))
                    cur_v = s.get("current_value")
                    base_v = s.get("baseline_value")
                    dev = s.get("deviation_percent", 0.0)
                    z = s.get("z_score", s.get("zScore", 0.0))
                    val_str = f" (Observed: `{float(cur_v):.3f}`, Baseline: `{float(base_v):.3f}`)" if cur_v is not None and base_v is not None else ""
                    lines.append(f"- **{name}:** Deviation `{dev:+.2f}%`, z-score `{z:+.2f}`{val_str}")
            else:
                lines.append("- Multi-variable dynamic correlation anomaly across plant control loops.")
            lines.append(f"\n**Diagnostic Cause:** {root_cause} in subsystem `{subsystem}`.")
            return ("evidence_why", "\n".join(lines))

        # Intent: Which sensors are used in the system
        if has_system_sensors:
            lines = [
                f"### System Sensors Overview (Tennessee Eastman Plant Baseline)",
                "The Tennessee Eastman Process monitors **52 continuous process variables**:",
                "",
                "- **41 Process Measurement Variables (XMEAS 1–41):**",
                "  * Feed Streams: A Feed (Stream 1), D Feed (Stream 2), E Feed (Stream 3), Total Feed (Stream 4), Recycle Flow (Stream 8), Reactor Feed Rate (Stream 6).",
                "  * Reactor: Pressure, Temperature, Level.",
                "  * Separator: Temperature, Level, Pressure.",
                "  * Stripper: Level, Pressure, Temperature, Steam Flow.",
                "  * Other Operations: Compressor Work, Purge Gas Flow (Stream 9).",
                "  * Process Stream Compositions: Online analyzers on Streams 6, 9, and 11 (monitoring Components A through H).",
                "",
                "- **11 Manipulated Variables (XMV 1–11):**",
                "  * Feed Valves: D Feed, E Feed, A Feed, Total Feed.",
                "  * Recycle & Purge: Compressor Recycle Valve, Purge Valve.",
                "  * Column Valves: Separator Liquid Flow, Stripper Liquid Product, Stripper Steam Valve.",
                "  * Heat Transfer Loops: Reactor Cooling Water Flow Valve, Condenser Cooling Water Flow Valve.",
            ]
            top_sensors = evidence.get("top_anomalous_sensors", [])
            if top_sensors:
                s_names = ", ".join(s.get("display_name", s.get("name", "Sensor")) for s in top_sensors[:5])
                lines.append(f"\nIn this incident ({ev_id}), the active deviating sensors are: **{s_names}**.")
            return ("system_sensors", "\n".join(lines))

        # Intent: Sensor scores specifically
        if has_sensor_scores:
            top_sensors = evidence.get("top_anomalous_sensors", [])
            lines = [f"### Individual Sensor Scores & Deviations for Incident {ev_id}", ""]
            if top_sensors:
                for s in top_sensors:
                    name = s.get("display_name", s.get("name", "Unknown"))
                    dev = s.get("deviation_percent", 0.0)
                    z = s.get("z_score", s.get("zScore", 0.0))
                    trend = s.get("trend", "unknown")
                    contrib = s.get("contribution", 0.0)
                    cur_v = s.get("current_value")
                    base_v = s.get("baseline_value")
                    val_str = f", Observed = `{float(cur_v):.3f}`, Ideal = `{float(base_v):.3f}`" if cur_v is not None and base_v is not None else ""
                    lim = s.get("limit_context", "")
                    lim_text = f" -> **[Safety Envelope: {lim}]**" if lim else ""
                    lines.append(f"- **{name}:** z-score = `{z:+.2f}`, Deviation = `{dev:+.1f}%`, Trend = `{trend}`, Contribution score = `{contrib:.2f}`{val_str}{lim_text}")
            else:
                lines.append("No individual sensors showed severe univariate excursions; anomaly was detected via dynamic multi-sensor correlation.")
            return ("sensor_scores", "\n".join(lines))

        # Intent: Both detector scores and severity requested
        if has_detector_scores and has_severity:
            lines = [
                f"### Telemetry Scores & Severity for Incident {ev_id}",
                f"- **Assessed Severity:** **{sev}** (Evaluated against TEP process safety limits)",
                f"- **Overall Anomaly Score:** `{score:.3f}`",
                "",
                "**Detector Breakdown:**",
                f"- **LSTM Autoencoder:** Reconstruction error = `{l_sc:.4f}` (Threshold: `{l_th:.4f}`, Margin: `{l_ratio:.2f}x` — {l_alarm})",
                f"- **CVA Dynamics (Q-Statistic):** Score = `{q_sc:.1f}` (Threshold: `{q_th:.1f}`, Margin: `{q_ratio:.2f}x` — {q_alarm})",
                f"  * CVA Lag Metrics: Dominant lag `{dom_lag}`, Slow drift ratio `{slow_drift:.2f}`",
                "",
                f"- **Suspected Root Cause:** {root_cause} (`{subsystem}`)",
            ]
            return ("detector_severity", "\n".join(lines))

        # Intent: Detector Scores only
        if has_detector_scores:
            lines = [
                f"### Anomaly Detector Scores for Incident {ev_id}",
                f"- **Overall Anomaly Score:** `{score:.3f}`",
                "",
                "**Detector Metrics & Thresholds:**",
                f"- **LSTM Autoencoder:** Reconstruction error = `{l_sc:.4f}` (Threshold: `{l_th:.4f}`, Margin: `{l_ratio:.2f}x` — {l_alarm})",
                f"- **CVA Dynamics Q-Statistic:** Score = `{q_sc:.1f}` (Threshold: `{q_th:.1f}`, Margin: `{q_ratio:.2f}x` — {q_alarm})",
                f"- **CVA Lag Profile:** Dominant lag `{dom_lag}`, Slow drift ratio `{slow_drift:.2f}`",
            ]
            return ("detector_scores", "\n".join(lines))

        # Intent: Severity only
        if has_severity:
            sev_expl = {
                "CRITICAL": "Exceeds major safety thresholds (high risk of reactor trip, overpressure, or equipment shutdown).",
                "HIGH": "Significant process deviation approaching safety envelopes or severely impacting product specs.",
                "MEDIUM": "Moderate process disturbance; control loops are compensating but efficiency is degraded.",
                "LOW": "Minor drift within operational safety margins.",
            }.get(sev, "Evaluated against Tennessee Eastman Process safety limits.")
            lines = [
                f"### Severity Assessment for Incident {ev_id}",
                f"- **Assessed Severity Level:** **{sev}**",
                f"- **Safety Context:** {sev_expl}",
                f"- **Suspected Cause:** {root_cause} affecting `{subsystem}`.",
            ]
            return ("severity", "\n".join(lines))

        # Intent: Overview / What is this anomaly
        if has_overview:
            trig = det_ev.get("triggered_by", "fused")
            ctype = det_ev.get("change_type", "unspecified")
            top_s = evidence.get("top_anomalous_sensors", [])[:3]
            sensor_str = ", ".join(s.get("display_name", s.get("name", "Sensor")) for s in top_s) or "No single sensor dominating"
            lines = [
                f"### Incident Diagnosis for {ev_id}",
                f"- **Root Cause:** **{root_cause}**",
                f"- **Affected Subsystem:** `{subsystem}`",
                f"- **Severity:** **{sev}**",
                f"- **Alarm Trigger:** Mode `{trig}` ({ctype} dynamic pattern)",
                f"- **Primary Deviating Sensors:** {sensor_str}",
                "",
                f"**Recommended Action:** {action}",
            ]
            return ("overview", "\n".join(lines))

        # Intent: Sensors to inspect
        if has_sensors:
            top_sensors = evidence.get("top_anomalous_sensors", [])
            lines = [f"### Sensors to Inspect for Incident {ev_id}", ""]
            if top_sensors:
                for s in top_sensors:
                    name = s.get("display_name", s.get("name", "Unknown"))
                    dev = s.get("deviation_percent", 0.0)
                    z = s.get("z_score", s.get("zScore", 0.0))
                    trend = s.get("trend", "unknown")
                    cur_v = s.get("current_value")
                    base_v = s.get("baseline_value")
                    val_str = f" (Observed: `{float(cur_v):.3f}`, Ideal Baseline: `{float(base_v):.3f}`)" if cur_v is not None and base_v is not None else ""
                    lim = s.get("limit_context", "")
                    lim_text = f" -> **[Limit Warning: {lim}]**" if lim else ""
                    lines.append(f"- **{name}:** Deviation `{dev:+.1f}%`, z-score `{z:+.2f}`, Trend: `{trend}`{val_str}{lim_text}")
            else:
                lines.append("No individual sensors showed severe statistical excursions; anomaly was detected via dynamic correlation.")
            return ("sensors_inspect", "\n".join(lines))

        # Intent: Recommended action / mitigation / fix
        if has_action:
            lines = [
                f"### Recommended Actions & Mitigations for Incident {ev_id}",
                f"- **Direct Operator Action:** {action}",
                f"- **Subsystem Target:** Inspect all valves and instruments in `{subsystem}`.",
                f"- **Safety Precaution:** {uncertainty}",
            ]
            return ("action", "\n".join(lines))

        # Generic why/explain fallback -> overview (only when nothing else hit,
        # so specific metric/evidence/severity intents keep priority).
        if has_generic_why:
            trig = det_ev.get("triggered_by", "fused")
            ctype = det_ev.get("change_type", "unspecified")
            top_s = evidence.get("top_anomalous_sensors", [])[:3]
            sensor_str = ", ".join(s.get("display_name", s.get("name", "Sensor")) for s in top_s) or "No single sensor dominating"
            lines = [
                f"### Incident Diagnosis for {ev_id}",
                f"- **Root Cause:** **{root_cause}**",
                f"- **Affected Subsystem:** `{subsystem}`",
                f"- **Severity:** **{sev}**",
                f"- **Alarm Trigger:** Mode `{trig}` ({ctype} dynamic pattern)",
                f"- **Primary Deviating Sensors:** {sensor_str}",
                "",
                f"**Recommended Action:** {action}",
            ]
            return ("overview_fallback", "\n".join(lines))

        return None

    @staticmethod
    def _render_event_context(event: Dict, previous_report: Dict) -> str:
        if not isinstance(event, dict):
            event = {}
        if not isinstance(previous_report, dict):
            previous_report = {}
        evidence = event.get("evidence", {}) if isinstance(event.get("evidence"), dict) else {}
        det_ev = evidence.get("detector_evidence", {}) if isinstance(evidence.get("detector_evidence"), dict) else {}
        sev = str(previous_report.get("severity") or evidence.get("severity", "unknown")).upper()
        score = float(event.get("max_anomaly_score", evidence.get("anomaly_score", 0.0)))
        ev_id = event.get("event_id", "UNKNOWN")
        root_cause = previous_report.get("root_cause", "Unknown")
        subsystem = previous_report.get("affected_subsystem") or evidence.get("candidate_subsystem", "Unknown")
        rec_action = previous_report.get("recommended_action") or "Inspect suspect equipment on site and verify control loops."
        uncertainty = previous_report.get("uncertainty") or "Verify on-site sensor calibrations."

        lines = [
            f"### Incident Diagnostic Summary",
            f"- Event ID: {ev_id}",
            f"- Assessed Severity: {sev}",
            f"- Suspected Root Cause: {root_cause}",
            f"- Affected Subsystem: {subsystem}",
            f"- Recommended Action / Fix: {rec_action}",
            f"- Precaution: {uncertainty}",
            "",
            f"### Overall Anomaly Detector Scores (Model-Level Metrics)",
            f"- Overall Combined Anomaly Score: {score:.3f}",
        ]

        # Detector breakdown
        trig = det_ev.get("triggered_by", "fused")
        ctype = det_ev.get("change_type", "unspecified")
        lines.append(f"- Alarm Trigger: {trig} mode ({ctype} dynamic pattern)")

        lstm = det_ev.get("lstm_ae") if isinstance(det_ev.get("lstm_ae"), dict) else {}
        if lstm and "score" in lstm:
            l_sc = float(lstm.get("score", 0.0))
            l_th = float(lstm.get("threshold", 0.0))
            l_ratio = l_sc / max(l_th, 1e-9)
            is_l_alarm = bool(lstm.get("alarmed") or lstm.get("is_anomalous") or (l_sc >= l_th and l_th > 0))
            l_alarm = f"ALARMED (Active — {l_ratio:.2f}x threshold)" if is_l_alarm else "Normal (Below limit)"
            lines.append(f"- LSTM Autoencoder: reconstruction error = {l_sc:.4f} (threshold = {l_th:.4f}, margin = {l_ratio:.2f}x, status: {l_alarm})")

        dyn = det_ev.get("dynamic") if isinstance(det_ev.get("dynamic"), dict) else {}
        cva = dyn.get("cva") if isinstance(dyn.get("cva"), dict) else {}
        if cva and "score" in cva:
            q_sc = float(cva.get("score", 0.0))
            q_th = float(cva.get("threshold", 0.0))
            q_ratio = q_sc / max(q_th, 1e-9)
            is_q_alarm = bool(cva.get("alarmed") or cva.get("is_anomalous") or (q_sc >= q_th and q_th > 0))
            q_alarm = f"ALARMED (Active — {q_ratio:.2f}x threshold)" if is_q_alarm else "Normal (Below limit)"
            lines.append(f"- CVA Dynamics Q-Statistic: score = {q_sc:.1f} (threshold = {q_th:.1f}, margin = {q_ratio:.2f}x, status: {q_alarm})")
            if cva.get("dominant_lag") is not None:
                lines.append(f"  * CVA lag metrics: dominant lag = {cva.get('dominant_lag')}, slow drift ratio = {float(cva.get('slow_drift_ratio', 0.0)):.2f}")

        # Top sensor deviations
        lines.append("")
        lines.append("### Individual Sensor Values, Baselines & Deviations (Sensor Telemetry)")
        top_sensors = evidence.get("top_anomalous_sensors", [])
        if isinstance(top_sensors, list) and top_sensors:
            for s in top_sensors:
                if not isinstance(s, dict):
                    continue
                name = s.get("display_name", s.get("name", "Unknown"))
                tag = s.get("name", "")
                tag_str = f" [{tag}]" if tag and tag != name else ""
                cur_v = s.get("current_value")
                base_v = s.get("baseline_value")
                z = s.get("z_score", s.get("zScore", 0.0))
                dev = s.get("deviation_percent", 0.0)
                trend = s.get("trend", "unknown")
                contrib = s.get("contribution", 0.0)
                lim = s.get("limit_context", "")
                lim_str = f" -> [Safety Limit: {lim}]" if lim else ""

                if cur_v is not None and base_v is not None:
                    lines.append(
                        f"- {name}{tag_str}: Observed Value = {float(cur_v):.4f}, "
                        f"Ideal/Normal Baseline = {float(base_v):.4f}, "
                        f"Deviation = {dev:+.2f}%, z-score = {z:+.2f}, Trend = {trend}, "
                        f"Contribution = {contrib:.3f}{lim_str}"
                    )
                else:
                    lines.append(
                        f"- {name}{tag_str}: Deviation = {dev:+.2f}%, z-score = {z:+.2f}, "
                        f"Trend = {trend}, Contribution = {contrib:.3f}{lim_str}"
                    )
        else:
            lines.append("- No individual sensors showed severe statistical excursions; anomaly was detected via multi-sensor correlation.")

        # Temporal sequence if present (handles both dict and list representations)
        raw_seq = evidence.get("temporal_sequence")
        if isinstance(raw_seq, dict):
            seq = [item for item in raw_seq.get("sequence", []) if isinstance(item, dict)]
        elif isinstance(raw_seq, list):
            seq = [item for item in raw_seq if isinstance(item, dict)]
        else:
            seq = []

        if seq:
            lines.append("")
            lines.append("### Event Timeline & Propagation Sequence")
            for i, item in enumerate(seq, start=1):
                dt = float(item.get("relative_time_minutes", 0.0))
                tag = "initiating disturbance" if i == 1 and dt == 0.0 else f"+{dt:.1f} min cascade"
                dname = item.get("display_name") or item.get("name", "Sensor")
                lines.append(f"{i}. {dname} became anomalous ({tag})")

        # System Sensor Baseline Context (52 Total TEP Sensors)
        lines.append("")
        lines.append("### System Sensors Overview (Tennessee Eastman Plant Baseline)")
        lines.append(
            "- Total plant instrumentation: 52 continuous process variables.\n"
            "- 41 Process Measurement Variables (XMEAS 1 to 41): includes A Feed (Stream 1), D Feed (Stream 2), E Feed (Stream 3), "
            "Total Feed (Stream 4), Recycle Flow (Stream 8), Reactor Feed Rate (Stream 6), Reactor Pressure, Reactor Temperature, "
            "Reactor Level, Separator Temperature, Separator Level, Separator Pressure, Stripper Level, Stripper Pressure, "
            "Stripper Temperature, Compressor Work, Purge Gas Flow (Stream 9), and Component Compositions (Streams 6, 9, 11).\n"
            "- 11 Manipulated Variables (XMV 1 to 11): D Feed Valve, E Feed Valve, A Feed Valve, Total Feed Valve, Compressor Recycle Valve, "
            "Purge Valve, Separator Liquid Flow Valve, Stripper Liquid Product Valve, Stripper Steam Valve, Reactor Cooling Water Flow Valve, "
            "Condenser Cooling Water Flow Valve."
        )

        return "\n".join(lines)