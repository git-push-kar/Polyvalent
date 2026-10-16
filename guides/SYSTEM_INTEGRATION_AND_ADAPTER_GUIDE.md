# End-to-End System Integration, Pipeline Architecture, and Adapter Interoperability Guide

**Target System:** Industrial Cyber-Physical Process Monitoring & Grounded Diagnostic Framework  
**Primary Reference Artifact:** Tennessee Eastman Process (TEP) Neuro-Symbolic Dual-Brain Engine  
**Document Type:** Technical Integration Architecture, Multi-Model Adaptation, and Enterprise Deployment Manual  
**Repository Source:** `C:\Users\Admin\Desktop\anomaly`  

---

## Executive Summary

This document specifies the complete engineering architecture, dataflow pipelines, interface contracts, and model adaptation procedures for the **Neuro-Symbolic Dual-Brain Industrial Anomaly Detection and Root Cause Analysis (RCA) System**. 

The system decouples high-frequency numerical time-series perception from symbolic language reasoning:
1. **Layer A (Sensor Brain - System 1 Intuition):** A real-time, zero-leakage unsupervised perception engine combining a spatial-temporal Long Short-Term Memory Autoencoder (LSTM-AE) with an analytical Canonical Variate Analysis (CVA) / Dynamic Principal Component Analysis (DPCA) Hankel subspace projection head.
2. **Evidence Extraction Bridge:** A deterministic translation layer that converts high-dimensional tensor anomalies into an auditable, symbolic evidence graph (calibrated $z$-scores, steady-state percentage drifts, chronological onset cascade ordering, and chemical subsystem safety boundaries).
3. **Layer B (Language Brain - System 2 Deliberation):** A parameter-efficient fine-tuned (PEFT) foundation model adapter (`tep_rca` / `intern_fused_lr`) that consumes structured evidence tokens to deliver deterministic diagnostic verdicts, thermodynamic reasoning, and actionable mitigation protocols.

This guide provides step-by-step instructions for:
* **Integrating the complete pipeline into external SCADA / DCS / Industrial IoT infrastructures.**
* **Decoupling and porting the trained LoRA adapter to alternative foundation backbones** (e.g., LLaMA-3, Qwen-2.5, Mistral, Gemma, or larger InternVL variants).
* **Merging and exporting weights for high-throughput serving** (via vLLM, Ollama, TensorRT-LLM, or GGUF).
* **Transferring the neuro-symbolic methodology to other industrial domains** (semiconductor manufacturing, power generation, wastewater treatment, data center telemetry, and robotics fleets).

---

## 1. System Architecture & Pipeline Dataflow

### 1.1 High-Level End-to-End Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SCADA / DCS SENSOR INGESTION LAYER                               │
│                                                                                                  │
│   52 Raw Industrial Telemetry Channels (Sampling Interval: Ts = 3 min, Latency < 1.0 ms)        │
│   [XMEAS(1..41) Process Measurements, XMV(1..11) Actuator / Valve Position Feedback Signals]     │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    PREPROCESSING & WINDOWING                                     │
│                                                                                                  │
│   • Non-Contaminating StandardScaler: z_t = (x_t - μ_nominal) ⊘ σ_nominal                        │
│   • Temporal Ring Buffer: W = 60 samples (3.0 hours history), Stride s = 5 samples               │
│   • Window Tensor Shape: X_k ∈ ℝ^(60 × 52)                                                       │
└───────────────────────────────────────┬───────────────────────────────┬──────────────────────────┘
                                        │                               │
                                        ▼                               ▼
┌───────────────────────────────────────────────┐ ┌────────────────────────────────────────────────┐
│         HEAD 1: DEEP LSTM AUTOENCODER         │ │       HEAD 2: DYNAMIC SUBSPACE CVA HEAD        │
│                                               │ │                                                │
│  • Encoder: 2-Layer LSTM (H=128, Bottleneck=64│ │  • Past/Future Hankel Form: p=5, f=5 (dim 260) │
│  • Decoder: 2-Layer LSTM (H=128)              │ │  • SVD Decomposition: H = U Γ V^T (r=20 states)│
│  • Metric: MSE Reconstruction Error S_LSTM    │ │  • Metric: Residual Q-statistic with EWMA      │
│  • Static Level Shifts & Catastrophic Trips   │ │  • Dynamic Oscillations & Valve Stiction       │
└───────────────────────┬───────────────────────┘ └────────────────────────┬───────────────────────┘
                        │                                                  │
                        └───────────────────────┬──────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         NORMALIZED OR-FUSION & EVENT AGGREGATION ENGINE                          │
│                                                                                                  │
│   • Normalized Score: S_fused = max( S_LSTM / θ_LSTM ,  S_CVA / θ_CVA ) * θ_LSTM                 │
│   • False Alarm Rate Constraint: FAR ≤ 0.01 on normal operational validation partitions         │
│   • 3-Window Confirmation State Machine: Requires 3 consecutive anomalous windows to fire        │
│   • Hysteresis Reset: Requires 20 consecutive normal windows to mark incident closed             │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼ [Incident Confirmed]
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              NEURO-SYMBOLIC EVIDENCE EXTRACTION BRIDGE                           │
│                                                                                                  │
│   • Sensor Contribution: MSE decomposition & z-score sorting (z_j = (x_j - μ_j) / σ_j)           │
│   • Percentage Drift: Δ%_j = ((x_current,j - baseline_j) / baseline_j) * 100                     │
│   • Temporal Onset Cascade: t_onset(j) = min{ t | |e_t,j| ≥ 1.5 σ_j }                            │
│   • Process Subsystem Grouping: Heuristic voting (feed, reactor, cooling, separator, stripper)   │
│   • Physical Safety Envelopes: Pressure (> 2850 kPa), Runaway Temperature (> 125 °C), Stiction   │
│   • Output Artifact: Validated AnomalyEvent Dictionary (JSON)                                    │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                LAYER B: GROUNDED LANGUAGE ADAPTER                                │
│                                                                                                  │
│   • Foundation Model: InternVL3-2B-LR / InternVL2-2B (2.2B Parameters, Frozen)                   │
│   • Vision Bypass: Dummy image tensor [1, 3, 448, 448] with image_flags = 0                      │
│   • Parameter-Efficient Adapter: tep_rca / intern_fused_lr (LoRA Rank-16, ~24M params)          │
│   • Generation Engine: Manual greedy decoding through forward pass with KV cache prefill         │
│   • Post-Generation Guardrail: Zero-hallucination telemetry scrubbing and schema validation      │
└───────────────────────────────────────┬───────────────────────────────┬──────────────────────────┘
                                        │                               │
                                        ▼                               ▼
┌───────────────────────────────────────────────┐ ┌────────────────────────────────────────────────┐
│          AUTOMATIC ROOT-CAUSE REPORT          │ │         MULTI-TURN OPERATOR DIALOGUE           │
│                                               │ │                                                │
│  Emitted instantaneously upon event trigger:  │ │  Interactive QA with control room engineers:   │
│  • Root cause identification                  │ │  • "What caused the pressure spike?"           │
│  • Affected subsystem & severity rating       │ │  • "Compare cooling water flow to setpoint"    │
│  • 3-bullet Chain-of-Thought reasoning steps  │ │  • Telemetry rendered in markdown tables       │
│  • Verifiable quoted telemetry evidence       │ │  • Strict numeric grounding enforcement        │
│  • Concrete mitigation protocol               │ │  • Zero out-of-thin-air hallucinations         │
└───────────────────────────────────────────────┘ └────────────────────────────────────────────────┘
```

---

## 2. In-Depth Component Specifications & Code Walkthrough

### 2.1 Layer A: Sensor Perception Brain

The perception layer is encapsulated within `AnomalyDetector` (`anomaly_detection/inference.py`) and `DynamicDetectorRunner` (`anomaly_detection/dynamic/runner.py`, CVA in `anomaly_detection/dynamic/cva.py`).

#### 2.1.1 Data Preprocessing & Non-Contaminating Scaler
* **Implementation:** [`preprocessing/scaler.py`](file:///C:/Users/Admin/Desktop/anomaly/preprocessing/scaler.py)
* **Mathematical Contract:**
  $$\tilde{\mathbf{x}}_t = (\mathbf{x}_t - \boldsymbol{\mu}_{\text{train}}) \oslash \boldsymbol{\sigma}_{\text{train}}$$
  Scalers are strictly fitted on the 350 normal training runs and serialized to `outputs/preprocessing/scaler.pkl`. Under no circumstances are test or fault data included during fitting.
* **Sliding Window Generation:** [`preprocessing/windowing.py`](file:///C:/Users/Admin/Desktop/anomaly/preprocessing/windowing.py) buffers samples into slices of dimension $(60, 52)$ with a step stride of 5 samples.

#### 2.1.2 Dual-Head Execution Details
* **LSTM Autoencoder Head:** Defined in `anomaly_detection/lstm_autoencoder.py`. Uses an encoder-decoder LSTM structure with a 64-dimensional bottleneck. Global MSE error is compared against $\theta_{\text{LSTM}} = 0.687363$.
* **Canonical Variate Analysis (CVA) Dynamic Head:** Defined in `anomaly_detection/dynamic/cva.py`. Constructs past vector $\mathbf{p}_t \in \mathbb{R}^{260}$ and future vector $\mathbf{f}_t \in \mathbb{R}^{260}$ ($p=5, f=5$). Performs SVD on the normalized Hankel cross-covariance matrix $\mathbf{H}$ to extract $r=20$ state variables. Dynamic residuals are monitored via squared prediction error $Q_t = \mathbf{e}_t^T \mathbf{e}_t$ and smoothed using recursive EWMA:
  $$\bar{Q}_t = 0.9 \bar{Q}_{t-1} + 0.1 Q_t$$
  Alarmed if $\bar{Q}_t > \theta_{\text{CVA}} = 239.75$ (Q-statistic, `outputs/anomaly_detector/dynamic/dynamic_thresholds.json`).

#### 2.1.3 Event State Machine
Implemented in `main.py:TEPApp` with persistence in `events/event_store.py` (SQLite `outputs/events.db`):
* Prevents transient single-window noise from firing alarms.
* Requires **3 consecutive anomalous windows** to instantiate a new `AnomalyEvent`.
* Maintains an active event until **20 consecutive normal windows** confirm system recovery.
* Automatically assigns a unique tracking identifier (e.g., `ANOM-DIST0208`).

---

### 2.2 The Evidence Extraction Bridge

The bridge is implemented in [`evidence/event_builder.py`](file:///C:/Users/Admin/Desktop/anomaly/evidence/event_builder.py), [`evidence/sensor_contribution.py`](file:///C:/Users/Admin/Desktop/anomaly/evidence/sensor_contribution.py), and [`evidence/process_limits.py`](file:///C:/Users/Admin/Desktop/anomaly/evidence/process_limits.py).

#### Symbolic Data Contract (Emitted Event Schema)
When an event triggers, Layer A emits a structured evidence dictionary conforming to the following JSON schema:

```json
{
  "event_id": "ANOM-1042",
  "anomaly_score": 0.942,
  "threshold": 0.687,
  "detector_type": "fused_lstm_cva",
  "severity": "critical",
  "pre_anomaly_context": {
    "duration_minutes": 15.0,
    "status": "normal"
  },
  "top_anomalous_sensors": [
    {
      "name": "XMV_11",
      "display_name": "Condenser_Cooling_Water_Flow",
      "current_value": 18.42,
      "baseline_value": 31.85,
      "deviation_percent": -42.17,
      "z_score": -3.84,
      "trend": "decreasing",
      "contribution": 0.384
    },
    {
      "name": "XMEAS_11",
      "display_name": "Separator_Temperature",
      "current_value": 94.61,
      "baseline_value": 80.12,
      "deviation_percent": 18.09,
      "z_score": 2.45,
      "trend": "increasing",
      "contribution": 0.261
    }
  ],
  "temporal_sequence": {
    "sequence": [
      {
        "sensor": "XMV_11",
        "display_name": "Condenser_Cooling_Water_Flow",
        "event": "decreased",
        "relative_time_minutes": 0.0
      },
      {
        "sensor": "XMEAS_11",
        "display_name": "Separator_Temperature",
        "event": "increased",
        "relative_time_minutes": 6.0
      }
    ],
    "first_onset_minutes": 0.0
  },
  "candidate_subsystem": "condenser_cooling_system",
  "candidate_subsystem_score": 0.92,
  "safety_limits_exceeded": [
    {
      "limit_name": "Condenser_Flow_Drop",
      "current": 18.42,
      "threshold": 22.0,
      "unit": "m3/hr"
    }
  ],
  "uncertainty": "Correlation does not prove causation; verify on-site valve positioner."
}
```

---

### 2.3 Layer B: Grounded Language Adapter (`tep_rca`)

The language layer is implemented in [`llm/model.py`](file:///C:/Users/Admin/Desktop/anomaly/llm/model.py), [`llm/adapter_loader.py`](file:///C:/Users/Admin/Desktop/anomaly/llm/adapter_loader.py), and [`llm/inference.py`](file:///C:/Users/Admin/Desktop/anomaly/llm/inference.py).

#### 2.3.1 LoRA Configuration & Parameter Footprint
The adapter attaches exclusively to the linear projection weights of the transformer language model:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # v0 CORRECTED 2026-10-15: Qwen2 backbone exposes only these 7.
    # Legacy names (wqkv/wo/w1/w2/w3) never materialized as weights
    # (verified: 392 safetensor keys = 7 modules x 28 layers x 2).
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM
)
```

| Component | Count | Parameter Share |
|---|---|---|
| **Base Model Parameters (InternVL3-2B fused, Qwen2)** | 2,218,452,992 | ~99.17% (Strictly Frozen) |
| **LoRA Trainable Parameters (v0 `intern_fused_lr`)** | ~18,500,000 | ~0.83% (7 modules x 28 layers x 2, r=16) |
| **Precision** | bfloat16 / fp16 | 74 MB safetensors footprint |
| **Comparison model v1** | same 7-module geometry | trained on fault-disjoint holdout (1944/36/36); see §6.5 of the research paper |

#### 2.3.2 Vision Bypass & Prompt Ingestion
InternVL's vision module is bypassed for sensor reasoning by passing a zero tensor with `image_flags = 0`.
The structured evidence dictionary is injected into the standard `internlm2-chat` prompt format:

```text
<|im_start|>system
You are an industrial process anomaly diagnosis assistant for the Tennessee Eastman Process (TEP)...
<|im_end|>
<|im_start|>user
[ANOMALY INCIDENT TELEMETRY]
Event ID: ANOM-1042
Severity: critical
Top Deviating Sensors:
- Condenser_Cooling_Water_Flow (XMV_11): observed 18.42 vs baseline 31.85 (-42.2%, trend: decreasing)
- Separator_Temperature (XMEAS_11): observed 94.61 vs baseline 80.12 (+18.1%, trend: increasing)
Onset Sequence:
1. Condenser_Cooling_Water_Flow at +0.0 min
2. Separator_Temperature at +6.0 min
Candidate Subsystem: condenser_cooling_system
Analyze this event and output a valid JSON report.
<|im_end|>
<|im_start|>assistant
```

#### 2.3.3 Output Schema & Guardrail Scrubbing
The adapter is trained to emit JSON containing ten mandatory fields:
1. `root_cause`: Grounded classification of the physical fault.
2. `affected_subsystem`: Physical plant area.
3. `severity`: `low` | `medium` | `high` | `critical`.
4. `confidence`: Float between $0.0$ and $1.0$.
5. `summary`: 1–2 sentence operator summary.
6. `reasoning_steps`: 3-bullet Chain-of-Thought (initiating sensor, physical loop mechanism, ruled-out alternatives).
7. `evidence`: Quoted list of observed telemetry values.
8. `reasoning`: Thermodynamic and chemical propagation analysis.
9. `recommended_action`: Concrete corrective protocol for control room operators.
10. `uncertainty`: Explicit statement on evidence bounds and verification advice.

**Grounding Guardrail:** [`llm/inference.py:sanitize_answer`](file:///C:/Users/Admin/Desktop/anomaly/llm/inference.py#L95-L120) programmatically intercepts generated JSON:
* Ensures all ten keys are populated.
* Cross-validates numbers against the input evidence block (flags deviations $>0.1\%$).
* Imputes deterministic severity bounds if the language model outputs `unknown`.

---

## 3. Step-by-Step Integration Guide: Using the Complete Pipeline

### 3.1 Python In-Memory API Integration (5 Lines of Code)

Any external application (e.g., streaming pipeline, Kafka consumer, Celery worker) can invoke the full system directly:

```python
from main import TEPApp
from utils import load_config

# 1. Initialize configuration and load models
config = load_config("configs/config.yaml")
app = TEPApp(config, enable_llm=True)

# 2. Process incoming 52-dimensional sensor telemetry stream
sensor_telemetry = {
    "values": [26.85, 3650.2, 4520.1, ...], # 52 floats
    "sample_index": 1204,
    "is_fault": True
}
result = app.process_sensor_stream(sensor_telemetry)

# 3. Check for confirmed anomaly and automatic RCA report
# (real keys: anomaly_detected/event_id/report/event/window_id/window_score/
#  is_anomalous/detector_evidence; +latency_ms via the REST layer)
if result.get("anomaly_detected"):
    print(f"Incident Triggered: {result['event_id']}")
    print(f"Root Cause: {result['report']['root_cause']}")
    print(f"Recommended Action: {result['report']['recommended_action']}")

# 4. Interactive Operator Follow-up Dialogue
answer = app.answer_followup(
    event_id="ANOM-1042", 
    question="Why did you suspect condenser cooling water instead of reactor cooling?"
)
print(answer["answer"])
```

---

### 3.2 REST API Server Integration

The repository includes a production-ready FastAPI microservice in [`api/server.py`](file:///C:/Users/Admin/Desktop/anomaly/api/server.py).

#### Starting the Server
```powershell
# Launch FastAPI backend on port 8000
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

#### API Route Reference Table

| Method | Endpoint | Description | Request Body / Parameters |
|---|---|---|---|
| `GET` | `/api/status` | System health, detector threshold, and adapter status | None |
| `POST` | `/api/stream/sample` | Ingests a single 52-sensor telemetry frame | `{"values": [...], "sample_index": 100}` |
| `POST` | `/api/stream/run` | Replays a full dataset file with simulated streaming | `{"source_file": "path.csv", "replay_rate": 10.0}` |
| `GET` | `/api/events` | Lists historical confirmed anomaly incidents | `?limit=50` |
| `GET` | `/api/events/{event_id}` | Retrieves detailed evidence graph and forensic report | Path parameter `event_id` |
| `POST` | `/api/events/{event_id}/followup` | Submits conversational operator questions | `{"question": "What is the status of XMV_11?"}` |

#### Ingesting Streaming Telemetry via cURL
```bash
curl -X POST "http://localhost:8000/api/stream/sample" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [0.25, 3600.1, 4500.0, 9.2, 26.8, 42.3, 2705.0, ...],
    "sample_index": 482,
    "is_fault": false
  }'
```

**Response:**
```json
{
  "anomaly_detected": false,
  "event_id": null,
  "report": null,
  "window_id": 11,
  "window_score": 0.612,
  "is_anomalous": false,
  "detector_evidence": {"triggered_by": "lstm_ae", "change_type": "level"},
  "open_event": false,
  "latency_ms": 0.842
}
```
Note: `api/server.py` currently calls `record.dict()` (pydantic v2 prefers
`model_dump()`); follow-up QA is intent-routed unless
`llm.followup.prefer_llm` is true (v0 gate restored it to `false`).

---

## 4. Adapter Interoperability: Porting to Other Models & Backbones

The `tep_rca` / `intern_fused_lr` adapter is an independent artifact containing only LoRA weight tensors. This section explains how to use it with other backbones or swap in other foundation models.

### 4.1 Scenario A: Loading the Adapter onto Other InternVL Variants

The v0 adapter attaches **only** to its exact training base (`./models/internvl3_2b_lr_fused`, Qwen2: hidden 1536 / inter 8960 / 28 layers). LoRA shapes are locked to those dimensions, so loading onto any other-sized backbone (`InternVL2-1B/4B/8B`, larger InternVL3 variants) fails with `RuntimeError: size mismatch`. Cross-backbone reuse = retrain per §4.2.

```python
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

base_model_path = "./models/internvl3_2b_lr_fused"  # v0's exact base — do NOT substitute sizes
adapter_dir = "outputs/intern_fused_lr_v0/intern_fused_lr"

# 1. Load scaled base model
model = AutoModel.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# 2. Attach the v0 TEP RCA LoRA adapter (same base only)
model = PeftModel.from_pretrained(
    model,
    adapter_dir,
    adapter_name="intern_fused_lr",
    is_trainable=False
)
model.eval()

# 3. Execute inference
# (v0 scores on its own distribution are in-distribution; see research paper §6.5.
#  The v1 holdout adapter in outputs/intern_fused_lr_v1/ is the comparison model
#  for zero-shot claims.)
```

---

### 4.2 Scenario B: Adapting Pure Text Foundation Models (LLaMA-3, Qwen-2.5, Mistral, Gemma)

If you prefer to deploy a pure text LLM (avoiding InternVL's dummy vision overhead), the modular training pipeline in [`llm/train_adapter.py`](file:///C:/Users/Admin/Desktop/anomaly/llm/train_adapter.py) can adapt any HuggingFace causal LM.

#### Advantages of Text-Only Backbones
* Eliminates the dummy image tensor (`dummy_image_size = 448`) and `image_flags = 0`.
* Eliminates the need to patch `img_context_token_id`.
* Allows native use of HuggingFace `model.generate()` with advanced KV-cache acceleration.

#### Step-by-Step Migration Code

```python
# configs/config_qwen.yaml
llm:
  base_model: "Qwen/Qwen2.5-7B-Instruct"  # Target text backbone
  adapter_name: "tep_rca_qwen"
  trust_remote_code: false
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  training:
    use_4bit: true
    num_epochs: 3
    learning_rate: 0.0002
    batch_size: 4
```

Train the adapter on our distilled telemetry dataset:
```powershell
python scripts/train_tep_adapter.py --config configs/config_qwen.yaml --output-dir outputs/qwen_adapter
```

Inference with the resulting text adapter:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id = "outputs/qwen_adapter/tep_rca_qwen"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id)

prompt = "<|im_start|>user\n[ANOMALY INCIDENT TELEMETRY]...\n<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs, 
    max_new_tokens=512, 
    temperature=0.1, 
    do_sample=False
)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

### 4.3 Scenario C: Permanent Weight Fusion (`merge_and_unload`) for High-Throughput Serving

To eliminate LoRA parameter overhead and serve the fine-tuned model via high-throughput inference engines (**vLLM**, **TGI**, **Ollama**, or **llama.cpp**), merge the adapter directly into the base model weights:

```python
"""scripts/export_merged_model.py: Merge LoRA weights into base model."""
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

base_path = "./models/internvl3_2b_lr_fused"
adapter_path = "outputs/intern_fused_lr_v0/intern_fused_lr"
export_path = "outputs/internvl_tep_standalone"

print(f"Loading base model from {base_path}...")
base_model = AutoModel.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"Attaching adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging LoRA weights permanently into base model...")
# W_new = W_0 + (alpha / r) * (A * B)
merged_model = model.merge_and_unload()

print(f"Saving standalone merged model to {export_path}...")
merged_model.save_pretrained(export_path, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tokenizer.save_pretrained(export_path)

print("Export completed successfully. Standalone model is ready for vLLM or Ollama conversion.")
```

#### Deploying with vLLM
Once merged, serve the model with continuous batching and PagedAttention:
```bash
vllm serve outputs/internvl_tep_standalone \
  --port 8080 \
  --trust-remote-code \
  --max-model-len 2048 \
  --dtype float16
```

---

### 4.4 Scenario D: Multi-LoRA Serving in Industrial Control Rooms

In complex production environments, multiple specialized LoRA adapters can be dynamically hot-swapped on a single shared base model instance:

```
                          ┌───────────────────────────┐
                          │   FROZEN BASE LLM         │
                          │   (InternVL / Qwen / Llama)│
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Adapter 1: tep_rca  │     │ Adapter 2: workorder │     │  Adapter 3: hazop    │
│  Root Cause Diagnosis│     │ SAP PM Work Orders   │     │ Safety Review & HAZOP│
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load single base model
base = AutoModelForCausalLM.from_pretrained("base_model_path", device_map="auto")
model = PeftModel.from_pretrained(base, "adapters/tep_rca", adapter_name="diagnosis")

# Attach auxiliary adapters
model.load_adapter("adapters/workorder_gen", adapter_name="workorder")
model.load_adapter("adapters/hazop_auditor", adapter_name="hazop")

# Dynamically switch adapters during execution
model.set_adapter("diagnosis")
diagnosis_report = model.generate(...)

model.set_adapter("workorder")
work_order_text = model.generate(...)
```

---

## 5. Domain Transfer Playbook: Adapting to Other Industrial Purposes

The neuro-symbolic dual-brain methodology is not restricted to the Tennessee Eastman Process. It directly generalizes to any high-dimensional multivariate cyber-physical system.

### 5.1 Overview of Transferable Industrial Domains

| Industrial Domain | Primary Sensors ($N$) | Physical Dynamics | Common Stealth Faults | Adaptation Strategy |
|---|---|---|---|---|
| **Semiconductor Etching** | 80–150 (RF power, chamber pressure, gas flows, optical emission) | Non-linear RF plasma interactions | Chamber wall polymer deposition, mass flow controller drift | CVA Hankel matrix on plasma emission lines; adapter trained on wafer scrap prevention |
| **Combined-Cycle Gas Turbines (CCGT)** | 120–200 (EGT thermocouple spreads, rotor vibration, compressor speeds) | High-speed rotational and thermal dynamics | Combustor flameout, blade fouling, sensor calibration drift | Dynamic PCA on exhaust gas temperature spread; adapter trained on thermal stress limits |
| **Water Treatment (SWaT / WADI)** | 51–68 (Flowmeters, pH, ORP, level switches, motor actuated valves) | Multi-stage hydraulic and chemical dosing | Cyber-physical spoofing, dosing valve stiction, filter clogging | LSTM-AE on tank levels + CVA on chemical dosing; adapter trained on EPA drinking water standards |
| **Data Center Server Fleets** | 40–100 per rack (Inlet/outlet air temp, fan RPM, GPU power, throttling) | Thermodynamic convective cooling and workload spikes | Airflow bypass, thermal runaway hotspots, VRM degradation | Spatial-temporal LSTM on temperature field; adapter trained on cooling optimization |

---

### 5.2 Step-by-Step Implementation Cookbook for New Domains

Follow this 5-stage procedure to instantiate the dual-brain system on a new industrial dataset:

```
  Stage 1: Ingest Normal Run Data & Compute Baselines
  Stage 2: Train Layer A Perception (LSTM-AE + CVA Hankel)
  Stage 3: Configure Domain Evidence Extraction Bridge
  Stage 4: Automated Teacher-Student Telemetry Distillation
  Stage 5: Train Domain LoRA Adapter & Deploy
```

#### Stage 1: Ingest Baseline Data & Compute Scaler
Collect 100+ hours of fault-free steady-state operation. Fit a non-contaminating scaler:
```python
from preprocessing.scaler import fit_scaler

# Compute empirical mean and standard deviation across normal operations
scaler = fit_scaler(normal_telemetry_dataframe, save_path="custom_domain/scaler.pkl")
```

#### Stage 2: Train Layer A Perception Heads
1. **Train LSTM-AE:** Run [`scripts/train_anomaly_detector.py`](file:///C:/Users/Admin/Desktop/anomaly/scripts/train_anomaly_detector.py) on normal windows. Calibrate threshold $\theta_{\text{LSTM}}$ to guarantee $\text{FAR} \le 0.01$.
2. **Train CVA Dynamic Subspace:** Run [`scripts/train_dynamic_detector.py`](file:///C:/Users/Admin/Desktop/anomaly/scripts/train_dynamic_detector.py). Set past/future horizons ($p=5, f=5$) and compute projection matrices $\mathbf{J}$ and $\mathbf{L}$. Calibrate threshold $\theta_{\text{CVA}}$.

#### Stage 3: Configure Domain Evidence Bridge
Define the domain dictionary in `custom_domain/knowledge.py`:
```python
SENSOR_METADATA = {
    "P_101": {"display_name": "Primary_Cooling_Pump_Pressure", "subsystem": "cooling", "unit": "bar"},
    "T_204": {"display_name": "Turbine_Inlet_Temperature", "subsystem": "turbine", "unit": "degC"},
    # ...
}

SAFETY_ENVELOPES = {
    "T_204": {"max_limit": 1150.0, "severity": "critical", "hazard": "Blade Thermal Overstress"},
    "P_101": {"min_limit": 2.5, "severity": "high", "hazard": "Coolant Cavitation Risk"}
}
```

#### Stage 4: Automated Distillation via Teacher LLM
Generate domain training samples using a high-capacity teacher model (e.g., `Qwen-2.5-32B` via Ollama):
```powershell
python scripts/generate_distillation_dataset.py `
  --teacher-model "qwen2.5:32b" `
  --api-base "http://localhost:11434" `
  --samples-per-fault 15 `
  --output-dir data/custom_domain_llm
```

#### Stage 5: Fine-Tune the Domain Adapter
```powershell
python scripts/train_tep_adapter.py `
  --config configs/custom_domain.yaml `
  --epochs 3 `
  --4bit `
  --output-dir outputs/custom_domain_adapter
```

---

## 6. Enterprise Deployment Architecture

### 6.1 Containerized Microservice Topology

For mission-critical on-premises deployment in industrial plants, the architecture packages into three independent Docker containers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 KUBERNETES / DOCKER POD                                │
│                                                                                        │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │ CONTAINER 1: INGESTION & PERCEPTION RUNNER (CPU Optimized)                     │   │
│   │ • Runtime: Python 3.11 + ONNX Runtime (CVA + LSTM-AE)                         │   │
│   │ • Throughput: > 1,500 samples/sec | Latency: 0.8 ms                           │   │
│   │ • Ingests SCADA Modbus / OPC-UA / MQTT broker streams                         │   │
│   │ • Evaluates 3-Window Aggregator and emits confirmed JSON events to IPC Socket  │   │
│   └───────────────────────────────────────┬────────────────────────────────────────┘   │
│                                           │ IPC / Redis Queue                          │
│                                           ▼                                            │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │ CONTAINER 2: REASONING & ADAPTER SERVICE (GPU Accelerated)                     │   │
│   │ • Runtime: FastAPI + PyTorch / vLLM                                           │   │
│   │ • Model: InternVL3-2B-LR + tep_rca (or Merged Standalone Engine)              │   │
│   │ • VRAM: 6–8 GB (4-bit QLoRA / bfloat16)                                       │   │
│   │ • Executes forensic report synthesis and multi-turn operator dialogue         │   │
│   └───────────────────────────────────────┬────────────────────────────────────────┘   │
│                                           │ REST / WebSocket                           │
│                                           ▼                                            │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │ CONTAINER 3: OPERATOR COCKPIT & HMI (Web Interface)                            │   │
│   │ • Runtime: Streamlit / React + Vite                                           │   │
│   │ • Live P&ID diagram animation, sensor trend charts, incident triage inbox     │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 6.2 Modbus TCP / OPC-UA Industrial Connector Example

Below is a complete industrial telemetry connector using Python's `pymodbus` / `opcua` library to bridge physical plant PLC signals into `TEPApp`:

```python
"""connectors/opcua_ingestion.py: Ingest physical DCS telemetry into TEPApp."""
import time
from opcua import Client
from main import TEPApp
from utils import load_config

OPC_ENDPOINT = "opc.tcp://192.168.1.100:4840"
CONFIG_PATH = "configs/config.yaml"

def run_telemetry_loop():
    config = load_config(CONFIG_PATH)
    app = TEPApp(config, enable_llm=True)
    
    client = Client(OPC_ENDPOINT)
    client.connect()
    print(f"Connected to Industrial OPC-UA Server at {OPC_ENDPOINT}")
    
    # Map 52 OPC Node IDs
    sensor_nodes = [client.get_node(f"ns=2;s=TEP.XMEAS_{i}") for i in range(1, 42)] + \
                   [client.get_node(f"ns=2;s=TEP.XMV_{i}") for i in range(1, 12)]
                   
    sample_idx = 0
    try:
        while True:
            t0 = time.time()
            # Read values synchronously from PLC
            raw_values = [node.get_value() for node in sensor_nodes]
            
            # Forward to Perception Engine
            out = app.process_sensor_stream({
                "values": raw_values,
                "sample_index": sample_idx
            })
            
            if out.get("anomaly_detected"):
                print(f"[ALARM] Incident {out['event_id']} confirmed!")
                print(f"Diagnosis: {out['report']['root_cause']}")
                # Optionally trigger DCS alarm bit
                # client.get_node("ns=2;s=TEP.Alarm_Trigger").set_value(True)
                
            sample_idx += 1
            # Maintain 3-minute industrial sample interval (or test rate)
            elapsed = time.time() - t0
            time.sleep(max(0.0, 3.0 - elapsed))
            
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_telemetry_loop()
```

---

## 7. Verification & Sanity Check Protocol

Before deploying the pipeline or an adapted model into production, execute this four-step test suite:

```powershell
# 1. Verify Unsupervised Anomaly Detection (Layer A)
python scripts/evaluate_anomaly_detector.py --config configs/config.yaml

# 2. Verify Dynamic CVA Head on Fault 15 (Valve Stiction)
python scripts/diagnose_fault15_full.py

# 3. Test Trained LoRA Adapter Loading & Inference Grounding (Layer B)
python scripts/test_adapter.py --config configs/config.yaml

# 4. Run Full End-to-End Streaming Simulation Integration Test
python scripts/test_end_to_end.py
```

### Expected Output Criteria
* **Layer A:** False Alarm Rate on normal holdout test runs must be $\le 1.5\%$. Fault 15 event detection rate must exceed $80\%$.
* **Layer B Adapter:**
  * Must load onto the base model with **0 unexpected prior adapters**.
  * Trainable parameter percentage must be $\approx 1.087\%$ (~24.1M parameters).
  * Emitted JSON must achieve 100% schema conformance with no unescaped strings or missing keys.
  * Quoted telemetry numbers must match the evidence block within $0.1\%$.

---

## 8. Summary of File & Artifact Locations

| Artifact | Repository Path | Description |
|---|---|---|
| **Base Model Weights** | `./models/internvl3_2b_lr_fused` | Frozen 2.2B vision-language foundation backbone |
| **Fine-Tuned Adapter v0 (live)** | `outputs/intern_fused_lr_v0/intern_fused_lr/` | Learned LoRA weights (`adapter_config.json` + `adapter_model.safetensors`), 7 Qwen2 modules |
| **Comparison Adapter v1** | `outputs/intern_fused_lr_v1/intern_fused_lr/` | Same geometry, trained on fault-disjoint holdout (zero-shot reference) |
| **LLM Data v0 / v1** | `data/llm/` / `data/llm_v1/` | v0: full-coverage in-distribution (2376/396/396); v1: holdout (1944/36/36) |
| **Preprocessing Scaler** | `outputs/preprocessing/scaler.pkl` | Non-contaminating normal training distribution mean/variance |
| **LSTM Autoencoder** | `outputs/anomaly_detector/model.pt` | Trained weights for static level-shift detection |
| **CVA Dynamic Matrices** | `outputs/anomaly_detector/dynamic/cva_model.npz` (+ `dpca_model.npz`, `dynamic_thresholds.json` with Q=239.75) | State projection matrix $\mathbf{J}$ and residual matrix $\mathbf{L}$ |
| **Distilled LLM Dataset**| `outputs/llm_dataset_v2/` | Distilled reasoning datasets generated via Qwen-2.5 |
| **SQLite Event Database**| `outputs/events.db` | Real-time persisted incident reports and conversation histories |
| **Unified Orchestrator**| [`main.py`](file:///C:/Users/Admin/Desktop/anomaly/main.py) | `TEPApp` single high-level interface class |
| **REST API Server**     | [`api/server.py`](file:///C:/Users/Admin/Desktop/anomaly/api/server.py) | Production FastAPI backend microservice |
| **Streamlit Dashboard** | [`ui/app.py`](file:///C:/Users/Admin/Desktop/anomaly/ui/app.py) | Interactive control room operator UI |
