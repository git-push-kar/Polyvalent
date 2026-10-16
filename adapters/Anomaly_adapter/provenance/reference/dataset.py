"""Supervised instruction-tuning dataset for InternVL2-2B (text-only).

The LLM does NOT receive raw sensor time series. It receives structured anomaly
events. Training examples follow the InternVL2 "internlm2-chat" prompt format
with the assistant answer being a JSON object, so the model learns to produce
the same structured output at inference time.

Text-only approach
------------------
TEP is sensor-only and contains no images. InternVL2-2B's ``forward()`` expects
a ``pixel_values`` tensor (it always runs the ViT), but the official InternVL
training code handles image-free samples with a dummy zero image and
``image_flags=0`` so no vision embeddings are injected. We follow exactly that
supported pattern rather than inventing an unsupported API.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"
EOS = "<|im_end|>"

SYSTEM_PROMPT = (
    "You are an industrial process anomaly diagnosis assistant for the "
    "Tennessee Eastman Process (TEP). You analyze structured anomaly evidence "
    "produced by an unsupervised sensor model. You reason about likely root "
    "causes, affected subsystems, severity, and recommended actions. You never "
    "claim absolute causal certainty: correlation and temporal order are "
    "supporting evidence, not proof. Whenever evidence is insufficient you say "
    "so explicitly and recommend verification.\n\n"
    "CHAIN-OF-THOUGHT INSTRUCTION: Think step-by-step before concluding. In your JSON "
    "output, first write 'reasoning_steps' breaking down: (1) which variable deviated first "
    "in the temporal sequence, (2) the direction and physical mechanism connecting the "
    "deviating sensors, and (3) why alternative subsystems are ruled out.\n\n"
    "SEVERITY RUBRIC:\n"
    "- critical: Risk of emergency trip/shutdown, vessel overpressure approaching safety limit (Reactor Pressure > 2900 kPa), "
    "reactor thermal runaway (> 125 °C), or unmitigated valve stiction / hysteresis in cooling loops (Faults 14, 15).\n"
    "- high: Large process excursions, product quality degradation, significant feed loss (A feed loss, C header pressure drop), "
    "or major cooling water step changes (Faults 4, 5, 6, 7, 11, 12, 21, 22).\n"
    "- medium: Moderate operational disturbances contained by regulatory control loops without safety hazard (feed ratio/composition "
    "step changes, feed temperature disturbances, random process variations) (Faults 1, 2, 3, 8, 9, 10, 16-20).\n"
    "- low: Minor transient variations or slow catalyst reaction kinetics drift without immediate operational risk (Fault 13)."
)

FOLLOWUP_SYSTEM_PROMPT = (
    "You are an industrial process assistant and control room diagnostic expert "
    "for the Tennessee Eastman Process (TEP). An operator is asking about a "
    "specific detected anomaly event. The user message contains the full "
    "incident telemetry (detector scores vs thresholds, sensor observed vs "
    "normal values with deviation %, temporal order) followed by the operator "
    "question.\n\n"
    "GROUNDING RULES:\n"
    "1. Answer using ONLY numbers present in the telemetry block. Quote exact "
    "detector scores/thresholds and sensor observed/baseline values.\n"
    "2. Never say 'as shown in the table below'. State the numbers directly.\n"
    "3. Name only sensors listed in the telemetry. Never invent sensor names.\n"
    "4. Keep temporal claims to the given propagation sequence; do not reorder it.\n"
    "5. If evidence is insufficient, say so and recommend on-site verification.\n\n"
    "FORMAT RULES (the chat UI renders markdown):\n"
    "- Score/value questions (metrics, evidence, sensor values): present numbers "
    "in markdown tables, one row per detector/sensor, then at most 2 short "
    "concluding lines. Tables use pipes, e.g.:\n"
    "| Detector | Score | Threshold | Status |\n"
    "|---|---|---|---|\n"
    "| LSTM reconstruction error | 0.6359 | 0.6874 | Normal (below limit) |\n"
    "| CVA Q-statistic | 1659.6 | 239.8 | ALARMED |\n\n"
    "| Sensor | Observed | Baseline | Dev% | z | Trend |\n"
    "|---|---|---|---|---|---|\n"
    "| Stripper_Level | 49.9542 | 50.0001 | -0.09% | -0.05 | stable |\n\n"
    "- Cause/severity/action questions: use short sections in this order, each "
    "1-3 bullets, no long paragraphs:\n"
    "**Verdict:** <one line>\n"
    "**Evidence:** <numbers only, table if more than 2 values>\n"
    "**Recommended action:** <concrete step>\n"
    "**Uncertainty:** <one line>\n"
    "- Never emit bare sensor lists like 'Name (TAG, ...)'. Every sensor row "
    "carries its numbers.\n\n"
    "VOICE AND STYLE (match a top-tier assistant: direct, structured, conversational-professional):\n"
    "- Lead with the direct answer in the first 1-2 sentences; no throat-clearing "
    "('The evidence is not sufficient...').\n"
    "- Commit to the single best-candidate cause with explicit confidence "
    "(high/medium/low); confine doubt to ONE closing uncertainty line plus verification. "
    "Reserve 'unclassified' for cases with no detector alarm and no sensor deviation.\n"
    "- Short markdown sections and bullets; paragraphs of 2-3 sentences. Explain the "
    "'why' mechanistically (initiating variable -> propagation) in your own words; "
    "never reuse stock sentences like 'The key point is...'."
)

ANSWER_KEYS = [
    "reasoning_steps", "summary", "root_cause", "affected_subsystem", "evidence",
    "reasoning", "severity", "confidence", "recommended_action", "uncertainty",
]

ANSWER_SCHEMA_HINT = (
    'Respond with ONLY a single JSON object using exactly these keys: '
    f'{", ".join(ANSWER_KEYS)}. severity must be one of '
    '"low", "medium", "high", "critical". confidence is a number between 0 and 1.'
)


def build_chat_prompt(
    system: str,
    messages: Sequence[Tuple[str, str]],
    assistant_turn: bool = True,
) -> str:
    """Build an internlm2-chat prompt with a custom system message.

    messages: list of (role, content) where role is "user" or "assistant".
    """
    parts = [f"<|im_start|>system\n{system}{EOS}"]
    for role, content in messages:
        if content:
            parts.append(f"<|im_start|>{role}\n{content}{EOS}")
        else:
            parts.append(f"<|im_start|>{role}\n")
    if assistant_turn and (not messages or messages[-1][1] is not None):
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def format_evidence_question(evidence: Dict, include_schema_hint: bool = True) -> str:
    """Render a structured event into the text prompt shown to the model."""
    from evidence.process_limits import compute_deterministic_severity, get_sensor_limit_context

    det_sev = evidence.get("severity") or compute_deterministic_severity(evidence)

    lines: List[str] = [
        "A new anomaly has been detected in the Tennessee Eastman Process.",
        "",
        f"Event ID: {evidence.get('event_id', 'UNKNOWN')}",
        f"Overall anomaly score: {evidence.get('anomaly_score', 0.0):.3f}",
        f"Assessed severity level: {det_sev} (evaluated against TEP process safety limits)",
    ]

    # Detector-specific scores and threshold ratios
    det_ev = evidence.get("detector_evidence", {}) or {}
    if det_ev:
        trig = det_ev.get("triggered_by", "unspecified")
        ctype = det_ev.get("change_type", "unspecified")
        lines.append(f"Detector trigger: {trig} mode ({ctype} dynamic pattern)")

        lstm = det_ev.get("lstm_ae") or {}
        if lstm and "score" in lstm:
            l_sc = float(lstm.get("score", 0.0))
            l_th = float(lstm.get("threshold", 0.0))
            l_ratio = l_sc / max(l_th, 1e-9)
            is_l_alarm = bool(lstm.get("alarmed") or lstm.get("is_anomalous") or (l_sc >= l_th and l_th > 0))
            lines.append(f"- LSTM Autoencoder: reconstruction error = {l_sc:.4f} (threshold = {l_th:.4f}, ratio = {l_ratio:.2f}x, alarmed = {is_l_alarm})")

        dyn = det_ev.get("dynamic") or {}
        cva = dyn.get("cva") or {}
        if cva and "score" in cva:
            q_sc = float(cva.get("score", 0.0))
            q_th = float(cva.get("threshold", 0.0))
            q_ratio = q_sc / max(q_th, 1e-9)
            is_q_alarm = bool(cva.get("alarmed") or cva.get("is_anomalous") or (q_sc >= q_th and q_th > 0))
            lines.append(f"- CVA Dynamics Q-Statistic: score = {q_sc:.1f} (threshold = {q_th:.1f}, ratio = {q_ratio:.2f}x, alarmed = {is_q_alarm})")
            if cva.get("dominant_lag") is not None:
                lines.append(f"  * CVA lag metrics: dominant lag = {cva.get('dominant_lag')}, slow drift ratio = {float(cva.get('slow_drift_ratio', 0.0)):.2f}")

    lines.append("")
    lines.append("Top sensor deviations:")
    sensors = evidence.get("top_anomalous_sensors", [])
    if sensors:
        for s in sensors:
            name = str(s.get("display_name", s.get("name", "")))
            z = s.get("z_score", s.get("zScore", 0.0))
            cur_v = s.get("current_value")
            base_v = s.get("baseline_value")
            val_str = f" [val={float(cur_v):.3f}, normal={float(base_v):.3f}]" if cur_v is not None and base_v is not None else ""
            line = (
                f"- {name}: "
                f"z={z:+.2f}{val_str} "
                f"(deviation: {s.get('deviation_percent', 0.0):+.1f}%, "
                f"trend: {s.get('trend', 'unknown')}, "
                f"contribution: {s.get('contribution', 0.0):.2f})"
            )
            lim = s.get("limit_context") or get_sensor_limit_context(name)
            if lim:
                line += f" -> [Operating Limit: {lim}]"
            lines.append(line)
    else:
        lines.append("- (no strong sensor deviations detected)")
    lines.append("")

    raw_seq = evidence.get("temporal_sequence")
    if isinstance(raw_seq, dict):
        seq = [item for item in raw_seq.get("sequence", []) if isinstance(item, dict)]
    elif isinstance(raw_seq, list):
        seq = [item for item in raw_seq if isinstance(item, dict)]
    else:
        seq = []

    if seq:
        lines.append("Temporal sequence (propagation timeline):")
        for i, item in enumerate(seq, start=1):
            dt = float(item.get("relative_time_minutes", 0.0))
            tag = "initiating disturbance" if i == 1 and dt == 0.0 else f"+{dt:.1f} min cascade"
            dname = item.get("display_name") or item.get("name", "Sensor")
            lines.append(
                f"{i}. {dname} became anomalous ({tag})."
            )
    else:
        lines.append("Temporal sequence: (no clear onset ordering detected)")
    lines.append("")

    subsystem = evidence.get("candidate_subsystem", "unknown")
    lines.append(f"Candidate affected subsystem: {subsystem}")

    context = evidence.get("pre_anomaly_context")
    if not isinstance(context, dict):
        context = {}
    lines.append(
        f"Pre-anomaly context: {context.get('status', 'unknown')} "
        f"(baseline duration ~{context.get('duration_minutes', '?')} min)"
    )
    if include_schema_hint:
        lines.append("")
        lines.append(
            "Determine the likely root cause and explain the evidence. "
            "Base your reasoning only on the evidence above. " + ANSWER_SCHEMA_HINT
        )
    return "\n".join(lines)


def render_answer_json(answer: Dict) -> str:
    """Serialize a structured answer dict to the exact JSON string trained on."""
    import json

    return json.dumps(answer, ensure_ascii=False)


class TEPInstructionDataset(torch.utils.data.Dataset):
    """Tokenized instruction samples for text-only InternVL2-2B SFT.

    Each sample yields: input_ids, attention_mask, labels (masked), a dummy
    ``pixel_values`` image and ``image_flags=0`` (official InternVL text-only
    pattern), so the model's ``forward()`` runs with no injected vision tokens.
    """

    def __init__(
        self,
        samples: Sequence[Dict],
        tokenizer: Any,
        max_length: int = 4096,
        dummy_image_size: int = 448,
        system_prompt: str = SYSTEM_PROMPT,
        pixel_dtype: Any = None,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.dummy_image_size = int(dummy_image_size)
        self.system_prompt = system_prompt
        self.pixel_dtype = pixel_dtype or torch.float32
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_parts(
        self, prompt: str, answer: str
    ) -> Tuple[List[int], List[int]]:
        """Tokenize prompt and answer separately and concatenate.

        The mask is applied from the assistant-turn header onward so only the
        answer tokens are learned (standard SFT masking).
        """
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=True,
            max_length=self.max_length,
        ).input_ids
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False, truncation=True,
            max_length=self.max_length,
        ).input_ids
        budget = self.max_length - len(prompt_ids)
        answer_ids = answer_ids[:budget]
        return prompt_ids, answer_ids

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        question = sample["question"]
        answer = sample["answer"]
        # Empty/missing system falls back to the default report prompt;
        # grounded_qa samples carry FOLLOWUP_SYSTEM_PROMPT instead.
        system = sample.get("system") or self.system_prompt

        prompt = build_chat_prompt(system, [("user", question)])
        prompt_ids, answer_ids = self._tokenize_parts(prompt, answer + EOS)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.zeros(
                1, 3, self.dummy_image_size, self.dummy_image_size,
                dtype=self.pixel_dtype,
            ),
            "image_flags": torch.tensor([0], dtype=torch.long),
        }


def _pad_batch(
    batch: List[Dict], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """Pad one collated batch to the longest sequence in it."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        ids, labs, attn = b["input_ids"], b["labels"], b["attention_mask"]
        pad = max_len - ids.shape[0]
        if pad > 0:
            ids = torch.cat([ids, torch.full((pad,), pad_token_id, dtype=ids.dtype)])
            labs = torch.cat([labs, torch.full((pad,), -100, dtype=labs.dtype)])
            attn = torch.cat([attn, torch.zeros(pad, dtype=attn.dtype)])
        out["input_ids"].append(ids)
        out["labels"].append(labs)
        out["attention_mask"].append(attn)
    out["input_ids"] = torch.stack(out["input_ids"])
    out["labels"] = torch.stack(out["labels"])
    out["attention_mask"] = torch.stack(out["attention_mask"])
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    out["image_flags"] = torch.stack([b["image_flags"] for b in batch])
    return out


class TEPCollator:
    """Picklable data collator (module-level class, safe for worker processes)."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return _pad_batch(batch, pad_token_id=self.pad_token_id)


def make_collator(pad_token_id: int):
    """Factory that injects the tokenizer pad id into the collator."""
    return TEPCollator(pad_token_id)
