# Deepfake Reasoning — Retrieval-Grounded Reasoning Plan

**Scope:** the *reasoning* half of the deepfake specialist only (the classifier
verdict path is already frozen). This is a design/plan document — no behavior
change beyond the already-landed streaming tag strip.

**Question answered:** *Can the deepfake reasoning be fed retrieved forensic
context the way RAG feeds a document, so the model then structures the whole
explanation around that context?*

**Answer:** yes — and the model is already trained for it. The reasoning LM is
conditioned on a short forensic-telemetry prefix built by
`build_grounding_prompt()` (`deepfake/wrapper.py`). What is missing is turning
that single hard-coded prefix into a proper retrieval step with a scored,
auditable context block. The plan below formalizes that.

---

## 1. Current state (already implemented, for reference)

| Piece | Where | Role |
| :--- | :--- | :--- |
| Classifier head | `cls_head` (`custom_heads.safetensors`) | `P(real)/P(fake)` from `[cls_token ‖ pooled ‖ freq_emb]` |
| Frequency branch | `freq_branch` + `compute_fft_features()` | 2D FFT log-magnitude → HF/LF energy ratio string |
| Attention hotspots | `AttentionPool` → `regions_from_topk()` | top-8 patch indices → named face regions |
| Telemetry template | `build_grounding_prompt()` (`wrapper.py:337`) | the **single** train/inference context format |
| Telemetry builder | `deepfake/forensics.py:build_telemetry_prompt()` | wraps the template for serving |
| Reasoning call | `predict(..., prompt_text=...)` (`wrapper.py:1943`) | inserts `vis_tokens + telemetry_embeds` before generation |
| Streaming path | `AdapterService.analyze_image_stream()` | classifier → telemetry → prompt → token stream (tags stripped for display) |

The grounding prefix already is a *retrieved context block*; it is just static
(three fields) and assembled inline rather than through a retriever. Training
uses the **same** template (`wrapper.py:1613`) with the true label, so the
continuation pattern is tightly coupled to this exact wording — any change
must be versioned and mirrored in both training and inference.

Current emitted context (verbatim shape):

```
[Forensic Telemetry]
- Statistical read-out (verify against what you see, do not blindly repeat): FAKE (confidence 98.7%)
- Frequency measurement: HF/LF log-magnitude energy ratio=0.48 (lower ratio = smoother image...)
- Attention hotspots: Forehead / Upper hairline, Nose & Cheek region

Weigh the telemetry numbers against the visual evidence, then explain using
<fast>, <planning>, <reasoning>, <reflection> and <conclusion> tags only.
Finish with exactly <answer>real</answer> or <answer>fake</answer> and nothing after it.
```

---

## 2. Target architecture: a retrieval-grounded forensic reasoner

```
 image ──► pixel_values (3,448,448) ─┐
       └─► freq_features (3,224,224) ┘
                                      │
        ┌─────────────────────────────▼─────────────────────────────┐
        │ STAGE 1  DETERMINISTIC RETRIEVAL (no model, no sampling)  │
        │  r1 classifier distribution  P(real), P(fake)            │
        │  r2 spectral: HF/LF ratio, radial energy curve           │
        │  r3 spatial: attention hotspot regions (top-k)           │
        │  r4 artifact KB: match telemetry signature → symptoms    │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  ranked, scored evidence items
        ┌─────────────────────────────▼─────────────────────────────┐
        │ STAGE 2  CONTEXT ASSEMBLY  (build_grounding_prompt v2)    │
        │  tokenize into known tag vocabulary; stable field order; │
        │  identical bytes at train + inference (versioned)        │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  telemetry prefix embeddings
        ┌─────────────────────────────▼─────────────────────────────┐
        │ STAGE 3  STRUCTURED GENERATION (Qwen2.5-1.5B + LoRA)      │
        │  <fast> <planning> <reasoning> <reflection> <conclusion> │
        │  final <answer>real|fake</answer>                        │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
        ┌─────────────────────────────▼─────────────────────────────┐
        │ STAGE 4  GUARDRAILS + PRESENTATION                        │
        │  numeric grounding check · answer-vocab check ·           │
        │  classifier arbitration · streaming tag strip             │
        └───────────────────────────────────────────────────────────┘
```

### Stage 1 — Deterministic retrieval

Retrieval here is **non-parametric and reproducible** (no vector DB needed at
first): every "document" is a computed forensic measurement or a KB entry keyed
by that measurement. This keeps the audit trail exact and avoids a second
model.

| id | Source | Computed by | Retrieved value |
| :--- | :--- | :--- | :--- |
| r1 | Classifier | `cls_head` + softmax | `verdict`, `confidence`, `P(real)`, `P(fake)` |
| r2 | Frequency | `batch_describe_frequency()` | HF/LF ratio; optionally per-radial-band energies |
| r3 | Spatial | `batch_regions_from_topk()` | top-2 named regions + raw top-k indices |
| r4 | Artifact KB | new `deepfake/artifact_kb.py` (proposed) | symptoms for signatures (generative over-smoothing, seam blur, compression grid, GAN fingerprint, warping/geometry) |
| r5 | Calibration | new `deepfake/calibration.py` (proposed) | temperature-scaled confidence vs. dataset base rates |

`r4` is the only genuinely new component: a small, human-curated table mapping
telemetry signatures to plain-language artifact descriptions. It is *lookup*,
not generation, so it is fully reproducible and citable. Example key:

```
signature: hf_lf_ratio < 0.60 AND hotspot in {"Forehead / Upper hairline",
           "Peripheral / Ear & Hair blending seams"}
→ symptom: "high-frequency attenuation at hairline seams — characteristic of
            generative over-smoothing / blend-seam softness"
```

### Stage 2 — Context assembly (the "RAG" block)

Extend `build_grounding_prompt()` to accept an ordered list of retrieved items
and render a **versioned** block. Two hard constraints:

1. **Byte-identical at train and inference** for the same inputs. Training
   already calls the same function (`wrapper.py:1613`); a `PROMPT_VERSION`
   constant must gate any change so old adapters never see a new wording.
2. **Known vocabulary only** — the fine-tuned continuation breaks on unseen
   prose, so retrieved KB text must be drawn from a fixed phrase set.

Proposed v2 render:

```
[Forensic Telemetry v2]
- Classifier: FAKE (confidence 98.7%; P_fake 0.987, P_real 0.013)
- Frequency: HF/LF log-magnitude energy ratio=0.48 (smoother than natural)
- Hotspots: Forehead / Upper hairline, Nose & Cheek region
- Artifact match: generative over-smoothing at hairline seams
Weigh every number above against what you see. Do not invent measurements.
Explain using <fast>, <planning>, <reasoning>, <reflection>, <conclusion>.
Finish with exactly <answer>real</answer> or <answer>fake</answer>.
```

### Stage 3 — Structured generation

Unchanged mechanically: `predict(prompt_text=...)` prepends
`vis_tokens + telemetry_embeds`, then decodes greedily (temperature 0) or
sampled. The tag vocabulary is part of the training contract, so the model
already emits the sectioned structure; the retrieved context simply gives it
more to structure *around*.

### Stage 4 — Guardrails + presentation

- **Numeric grounding:** every figure quoted in `<reasoning>` must appear in
  the context block (same idea as `llm/inference.py:sanitize_answer` in the
  anomaly project). Mismatch → flag `requires_human_review`.
- **Answer vocabulary:** the final `<answer>` must be exactly `real`/`fake`;
  otherwise fall back to the classifier label.
- **Arbitration:** classifier stays primary authority (see
  `INTEGRATION_GUIDE.md` §9); LLM/classifier disagreement sets the review flag.
- **Presentation:** tags are dropped from the user-facing stream, text kept
  (landed in `AdapterService.analyze_image_stream` + `api/server.py`).

---

## 3. Phased implementation plan

| Phase | Change | Files | Risk |
| :--- | :--- | :--- | :--- |
| P0 | *(done)* strip XML tags on the reasoning stream; keep text | `router/adapter_service.py`, `api/server.py` | none |
| P1 | Add `artifact_kb.py` (pure lookup) + `PROMPT_VERSION` gate; render v2 context | `deepfake/artifact_kb.py`, `deepfake/wrapper.py`, `deepfake/forensics.py` | low (display-only if version-gated) |
| P2 | Retrain/continue-train the deepfake LM LoRA on the v2 prompt so generation matches the richer context | `LY_C68/deepfake_project` training pipeline | medium (needs GPU + re-extract adapters) |
| P3 | Add numeric-grounding + answer-vocab guardrail to `analyze_image_stream`; surface `requires_human_review` | `deepfake/forensics.py`, `router/adapter_service.py` | low |
| P4 | Confidence calibration (temperature scaling) from the eval set | new `deepfake/calibration.py` | low |
| P5 | Optional true retriever (embedding index over past labeled cases) — only if P1-P4 prove the format stable | new `deepfake/retriever.py` | high (drift/leakage) |

**Recommendation:** ship P0 (done) and P1 now; P2 is required before enabling
the v2 context in production, because the adapter was trained on the v1 bytes.
P3 can land immediately (independent of prompt version). Defer P5.

---

## 4. Non-goals

- No change to the classifier heads or the frozen arbiter policy.
- No second LLM / judge model — retrieval stays deterministic.
- No free-form KB text until the LM has seen that vocabulary in training.

---

## 5. Provenance / mismatch notes

- Template source of truth is `deepfake/wrapper.py:build_grounding_prompt`
  (verbatim from `LY_C68/deepfake_project/model.py`). Both training and
  serving must import this one function — never re-implement it.
- Region names come from `REGION_NAMES` (6 fixed regions, grid 32×32); the KB
  must key off those exact strings.
- Frequency wording must stay compatible with
  `batch_describe_frequency()` output so the prompt remains byte-stable.
