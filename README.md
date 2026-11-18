# Polyvalent

One pristine **InternVL3-2B** base, several task specialists, and one router.
**Laya** (a System-1 decision model) reads each request and picks a task. The
service then loads **only the adapter that task needs** and releases the one
the previous task used.

| Task | What it does | Weights |
|---|---|---|
| `general` | Everyday chat, definitions, coding | bare base, no adapter |
| `reasoning` | Logic puzzles, deductions, "A or B?" decisions | LoRA r=64 on the base |
| `anomaly` | Tennessee Eastman (TEP) fault root-cause analysis | LoRA r=16 on the base |
| `deepfake` | Real or fake verdict for a face image, with forensic reasoning | dedicated wrapper (own backbone, LLM + vision LoRA, classifier heads) |

The web UI has three workspaces:

- **Chat**: Laya picks the base model, the reasoning adapter or the deepfake adapter for each message. Every reply shows which one answered and whether it was just loaded.
- **Anomaly detection**: replays a TEP test-set run, detects the fault, and asks the anomaly adapter for a root-cause report. You can ask follow-up questions about the incident.
- **Finance portfolio**: coming soon.

---

## Architecture

```
            ┌──────────────────────────────────────────────┐
request ──► │ Fine-tuned Laya router                       │
            │  (router/trained_router.py)                  │
            │  five-field state → questions.json model     │
            │  → BASE | DEEPFAKE                           │
            └───────────────┬──────────────────────────────┘
                            │ task
            ┌───────────────▼──────────────────────────────┐
            │ AdapterService.lease(task)                   │
            │  (router/adapter_service.py)                 │
            │   AdapterResidency.activate(task)            │
            │   (router/residency.py)                      │
            │     • load on first use, nothing at start-up │
            │     • ≤ 1 LoRA attached to the shared base   │
            │     • task change ⇒ evict old, attach new    │
            │     • deepfake ⇔ text tasks swap backbones   │
            └───────────────┬──────────────────────────────┘
                            │ resident model
                    token stream (SSE) ──► UI
```

### Routing (fine-tuned Laya)

Chat turns are routed by the checkpoint trained in the `Router` project
(`models/laya_router`, SHA-256 `cbca78ad…97e8`), loaded by
`router/trained_router.py`. It is asked the question the checkpoint was fitted
on — `BASE` or `DEEPFAKE` — over the five-field runtime state:

```json
{"message": "...", "assets": [], "active_image_id": null,
 "last_analysis": null, "history": []}
```

`router/conversations.py` builds that state server-side: real recent turns,
trusted attachment metadata (`id/kind/availability/support`), and the last
analysis in the archive's `{asset_id, confidence, evidence_summary, verdict}`
shape. Pixels never enter it, and `deep_thinking` is not part of it.

- `BASE` → the shared base model, with the actual image passed through
  InternVL's multimodal interface when one is attached.
- `DEEPFAKE` → the existing forensic pipeline (classifier verdict, telemetry,
  tagged reasoning), only when the active asset is available and supported.
- `deep_thinking=true` → the LR (reasoning) adapter directly; no router call.

There are no keyword rules, modality short-circuits or canned replies: an
attachment never bypasses the checkpoint, and "describe the person in this
image" is decided on the message, not on the presence of a file. Residual
context overflow drops the oldest history turns first (the message is
serialized first), and the receipt reports `history_dropped` rather than
silently losing the request.

After a deepfake run the verdict, score, hotspot/spectral evidence and the
user-visible explanation are stored against that conversation and image. A
follow-up such as "Explain that verdict" is routed again (normally `BASE`) and
receives the recorded analysis — not a second forensic pass; an explicit rerun
is what sends it back to `DEEPFAKE`.

Chat routes one image per turn. A current upload or image ID takes priority.
Within six transcript entries of an upload, an implicit follow-up can use the
active image. Later in the same conversation, an explicit reference such as
“the earlier portrait” or “the photo I uploaded” can still select that image.

For a follow-up with a stored image, the router evaluates the message with and
without the image metadata. The image-conditioned DEEPFAKE choice at probability
0.50 or above dispatches the selected image to the forensic adapter. Ordinary
visual questions can use the same stored pixels on BASE.

Because LAYA was fine-tuned on single messages and asset availability, routing
context sends `history: []` to LAYA to prevent cross-turn transcript pollution.
Generation context maintains the full persistent transcript.

### Multi-Face YOLO Detection & Context Cropping Pipeline

When an authenticity check (`DEEPFAKE`) runs on an uploaded image, the system
executes the YOLO multi-face pipeline (`deepfake/face_detect.py`, `models/yolov8n-face.pt`):

1. **Detection**: YOLO face detector identifies all face bounding boxes (letterbox 640, conf ≥ 0.25, NMS 0.50).
2. **Routing Contract**:
   - **Passthrough (Lone Portrait Face)**: If exactly 1 face is detected and covers ≥ 5% (0.05) of the frame area, it takes the untouched whole-image path (`inference.py` parity).
   - **Multi-Crop (`per_face`)**: If ≥ 2 faces are detected (or a lone face occupying < 5% of a wide scene), each face ≥ 96 px is cropped with a **30% background context ring** (`margin=0.30`) and squared. This retains hairline/jaw/neck blending seams and avoids synthetic texture.
   - **Fallback (`no_face`)**: If 0 faces are detected, the whole image is evaluated and flagged `faces_detected: 0`.
3. **Batched Classification & Telemetry**: Each face crop is preprocessed (448px visual tokens + 224px 2D FFT spectrum) and passed through the classifier head, computing per-face probabilities, confidences, attention hotspots, and spectral ratios.
4. **Threat Aggregation & Reasoning**:
   - Aggregation uses worst-case `max(p_fake)`: a single forged face renders the document `FAKE`.
   - The primary suspicious face (highest `p_fake`) is selected for in-depth, grounded LLM reasoning (`<fast>`, `<planning>`, `<reasoning>`, `<reflection>`, `<conclusion>`, `<answer>`).
5. **Persistence**: The full multi-face breakdown (`faces_detected`, `faces_analyzed`, per-face boxes and scores) is recorded in `Analysis` and included in SSE `verdict` and `analysis_saved` events, so subsequent follow-ups to BASE accurately reflect all evaluated faces.

The anomaly endpoint (`/anomaly/reason`) and the research endpoint
(`/route/laya/predict`) keep the four-way zero-shot `LayaRouter`
(`router/laya_router.py`) for `deepfake / anomaly / reasoning / general`; its
image short-circuit was removed so it also routes on the message.

Every routing receipt carries the serialized input, the checkpoint's route and
probabilities, and the token budget; the same data is written to
`logs/chat_runs.jsonl`.

### Adapter residency (lazy load and swap)

`router/residency.py` decides what is on the device:

| Request sequence | What happens |
|---|---|
| first `reasoning` | load base, attach `reasoning` |
| `reasoning` again | reuse (no I/O) |
| `anomaly` | detach `reasoning` (unwrap, **not** merged), attach `anomaly` |
| `general` | detach the LoRA; the bare base answers |
| `deepfake` (image) | detach the LoRA and release the shared base, then load the deepfake wrapper |
| any text task after that | release the deepfake wrapper and load the base again (plus the LoRA if needed) |

Two policies are available through `FINAL_ADAPTER_POLICY`:

- `swap` (default): only one backbone is on the GPU at a time. This is the safe choice for small GPUs.
- `keep`: once loaded, the shared base and the deepfake wrapper both stay resident. Use it on large GPUs. LoRAs are still attached one at a time.

Each request holds a **lease** on the model: swap, then generate, then release.
A second request cannot swap weights while a stream is running. It waits, up to
`FINAL_LEASE_TIMEOUT` seconds (600 by default), and then gets a "busy" notice.
If the client disconnects, generation stops and the lease is released.

A text-only "is this face fake?" request has no executable image, so the
checkpoint answers BASE and the base model replies that it cannot see one — no
forensic weights are loaded and the resident adapter stays where it is.

---

## Requirements

- Python 3.10+ (developed on 3.11)
- `fastapi`, `uvicorn`, `pydantic>=2`, `numpy`
- For real generation: `torch`, `transformers`, `peft`, `safetensors`, `pillow`, `torchvision`
- `laya` (the SDK that loads both Laya checkpoints). Without it the chat router cannot load: receipts say `gate-unavailable` and turns answer with the base model.
- Optional: `pytest` for the tests, `pyreadr` to split `.RData` TEP files

```powershell
pip install fastapi uvicorn numpy torch torchvision transformers peft safetensors pillow pytest
pip install laya   # or your internal wheel
```

## Adapters and paths

The weights live **inside this repository**: the base model in
`models/InternVL3-2B`, the task adapters in `adapters/`. They were copied from
the training runs, which keep their originals (`adapters/README.md` lists both
sides). Paths resolve from the repository root, so the service starts with no
environment variables; set any of them to point somewhere else.

| Variable | Default | Points to |
|---|---|---|
| `FINAL_BASE_DIR` | `models/InternVL3-2B` | pristine InternVL3-2B directory |
| `FINAL_ANOMALY_DIR` | `adapters/anomaly` | anomaly PEFT LoRA (`adapter_config.json` + `adapter_model.safetensors`) |
| `FINAL_REASONING_DIR` | `adapters/reasoning` | reasoning PEFT LoRA |
| `FINAL_DEEPFAKE_LLM_DIR` | `adapters/deepfake/llm_adapter` | deepfake LLM LoRA |
| `FINAL_DEEPFAKE_VISION_DIR` | `adapters/deepfake/vision_adapter` | deepfake vision LoRA |
| `FINAL_CUSTOM_HEADS` | `adapters/deepfake/custom_heads/custom_heads.safetensors` | deepfake classifier heads |

Other settings:

| Variable | Default | Meaning |
|---|---|---|
| `FINAL_ADAPTER_POLICY` | `swap` | `swap` or `keep` (see above) |
| `FINAL_LEASE_TIMEOUT` | `600` | seconds to wait for a busy model |
| `FINAL_LAYA_MIN_CONF` | `0.30` | Laya abstention threshold on max(p) |
| `FINAL_LAYA_ROUTER_DIR` | `models/laya_router` | fine-tuned BASE/DEEPFAKE Laya checkpoint (deepfake gate) |
| `FINAL_LAYA_ROUTER_DEVICE` | `cpu` | device for that checkpoint |
| `FINAL_STUB_GENERATION` | `0` | `1` = no weights; placeholder text, real routing and swap bookkeeping |
| `FINAL_TEP_TEST_DIR` | `data/tep_test` | TEP test runs for the anomaly tab |
| `FINAL_TEP_FAULT_ONSET` | `160` | sample index where faults start (both public test sets) |
| `FINAL_ANOMALY_REPO` | unset | path to the anomaly training repo, to use its trained LSTM-AE + CVA detector |
| `FINAL_ANOMALY_CONFIG` | `<repo>/configs/config.yaml` | config for that detector |
| `FINAL_ANOMALY_THRESHOLD` | `0.687363` | detector threshold shown in the UI |

## Running

```powershell
# real models (weights are in models/ and adapters/, no env vars needed)
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1

# UI/demo without weights
$env:FINAL_STUB_GENERATION = "1"
python -m uvicorn api.server:app --port 8000
```

Open <http://localhost:8000>. The tabs are at `#chat`, `#anomaly` and
`#finance`. The pill in the top-right corner shows what is loaded right now.

Use `--workers 1`. Each worker process would load its own copy of the weights.

## Anomaly detection tab

1. **Dataset.** Put TEP test runs in `data/tep_test/`, or set `FINAL_TEP_TEST_DIR`. Supported formats:
   - Braatz `d01_te.dat … d21_te.dat` (`d00_te.dat` is used as the fault-free baseline)
   - per-run CSVs with 52 sensor columns, with the fault number in the file name (`fault4_run1.csv`)
   - `.npy` arrays shaped (T, 52)

   **This repo ships the Rieth et al. (2017) TEP testing set**, split into per-run CSVs
   under `data/tep_test/`: `fault0_run{1,2,3}.csv` (fault-free) plus
   `fault01..fault20_run{1,2,3}.csv`, 960 samples x 52 sensors each, ~22 MB in total.
   They were split from
   `../anomaly/data/raw/normal/TEP_FaultFree_Testing.csv` and
   `../anomaly/data/raw/faults/TEP_Faulty_Testing.csv`; the source export is
   untouched. `data/tep_test/dataset.json` records that provenance and the fault
   onset (sample 20 = 1 h in this export, not the 8 h of the Braatz files).
   To split more runs from the same sources:
   ```powershell
   python scripts/prepare_tep_testset.py --src TEP_FaultFree_Testing.csv TEP_Faulty_Testing.csv --out data/tep_test --runs 1 2 3
   ```
   The four **synthetic** runs stay in the picker as well (labelled `[SYNTHETIC]`):
   they are the *inject a fault* option, the dataset runs are the real ones. They are
   not TEP physics. Fault onset comes from `FINAL_TEP_FAULT_ONSET` if set, else
   `dataset.json`, else 160.
2. **Detection and diagnosis.** `POST /anomaly/run` replays the run through the trained stack in `anomaly/` (`FINAL_ANOMALY_REPO` points at another checkout):
   - **Layer A detector** — that repo's `TEPApp`: normal-only scaler, LSTM-AE + CVA, event aggregation and the v2 evidence bridge.
   - **L1 diagnosis** — the hybrid in `outputs/analysis/phase3a/hybrid/vocabfix/protocolA`: 323 page features → LightGBM (15 classes) → isotonic calibration → menu rule over the 15 faults plus `UNCLASSIFIED_OPEN_SET` (`docs/reports/HYBRID_LIVE_INTEGRATION_PLAN_2026-11-16.md` §9). CPU only, needs `lightgbm`. `TEPApp` stores the verdict as `report["diagnosis"]`; the timeline returns it per event and as the top-level `diagnosis`. It never edits the prose — the label is its own field (§9.5).
   The stored operating point (0.687) is kept because it alarms on only 2.0 % of the bundled fault-free windows; if a checkout's `threshold.json` fails that check, the threshold is re-derived at p99 and `detector.threshold_source` records it. The menu threshold stays `threshold.json`'s 1.0 (a pre-registered false-abstention budget, §9.9); `FINAL_ANOMALY_MENU_THRESHOLD` pins another value.
   **Measured on all 60 test runs** (single-event replays, fault onset 20): 31 named correctly, 23 abstained, 4 raised no event, 2 confidently wrong (Faults 3 and 9 answered `NORMAL_TRANSIENT`); the classifier's top-1 was right in 36/60. Fault-free runs come back `NORMAL_TRANSIENT`/`no_fault` or no event. This is well below the 0.996 the same artifacts score on benchmark pages — those pages carry `pre_anomaly_context.post_mean`/`delta` maps that a 20-sample pre-fault window cannot produce, and their events are built differently. The §9.8 497-item reproducibility gate could not be re-run here because the stored validation rows carry predictions, not evidence pages.
3. **Evidence.** A confirmed event becomes the structured evidence dictionary described in the integration guide (§2.2): the top deviating sensors with observed and baseline values, deviation %, z-score and trend; the onset sequence; the candidate subsystem; safety limits; and severity.
4. **Reasoning.** `POST /anomaly/reason` sends the evidence through **Laya**. The sensor-vector modality routes it to `anomaly`, the anomaly LoRA is loaded if needed, and the adapter streams a JSON root-cause report using the training-time framing (`[ANOMALY INCIDENT TELEMETRY]` block). Follow-up questions use the follow-up system prompt with the same telemetry. The model only ever sees the evidence, never the raw time series.
5. **Comparison.** Tick "Show dataset fault label" to compare the diagnosis with the run's IDV label.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | web UI |
| POST | `/chat/stream` | end-user chat (SSE) |
| POST | `/route/laya/predict` | research endpoint with full routing and generation receipts (SSE) |
| GET | `/adapters/status` | what is loaded, the policy, and recent swaps |
| POST | `/adapters/release` | unload everything |
| GET | `/anomaly/scenarios` | available TEP runs |
| POST | `/anomaly/run` | `{scenario_id}` → detection timeline + events (JSON) |
| POST | `/anomaly/reason` | `{evidence, values?, question?, history?}` → RCA report or follow-up answer (SSE) |
| GET | `/route/stats`, `/health` | counters and health |

SSE events (one JSON object per `data:` line):

```
routing      {"router", "task", "reason", "input": <five-field state>, "output": <gate route/probabilities>, "input_tokens", "input_budget"}
route        {"task": "reasoning", "label": ...}           Laya's choice
context      {"history_turns", "analysis_id", "image_id", "analysis_supplied"}   what generation received
adapter      {"task", "loaded": [...], "unloaded": [...], "cold", "swap_ms"}
verdict      {"verdict": "REAL|FAKE", "confidence", "hotspots", "spectral"}   deepfake
analysis_saved {"analysis_id", "image_id", "verdict", "confidence"}          deepfake
thinking     {"text"}   deliberation (the UI shows it collapsed)
token        {"text"}   visible answer, streamed as it is generated
notice       {"text"}   UI feedback (unreadable image, busy, load failure); not part of the conversation
done         {"conversation_id"}
```

## Tests

```powershell
python -m pytest -q tests          # no weights needed (fake backend + stub mode)
```

- `tests/test_residency.py`: lazy loading, one LoRA at a time, backbone swaps, recovery from a failed attach, cross-thread lock release.
- `tests/test_api_stub.py`: chat task changes swap adapters, a text-only deepfake request does not evict the loaded adapter, the anomaly tab end to end, and disconnect cleanup.

Live-model regression scripts are in `scripts/`. For example,
`scripts/verify_chat_routing.py` and `scripts/verify_deepfake_live.py` load the
real weights, so stop the server before running them.

`scripts/verify_trained_routing.py` checks the five routing cases against the
fine-tuned checkpoint (no generation), and
`scripts/live_chat_trace.py` posts one turn to a running server and prints the
routing/context/adapter receipts.

## Repository layout

```
api/server.py            FastAPI app, SSE endpoints, lease handling
router/trained_router.py fine-tuned BASE/DEEPFAKE chat router (five-field state)
router/conversations.py  server-side turns, image assets, saved analyses
router/laya_router.py    four-way Laya router (anomaly + research endpoints)
router/residency.py      lazy load / swap policy (torch-free, unit-tested)
router/adapter_service.py  real backends (PEFT attach/unload, deepfake wrapper), streaming + image answers
router/prompts.py        training-faithful system prompts per task
deepfake/                ported DeepfakeReasoningModel wrapper, loader, forensics
tep_sim/                 TEP metadata, dataset loading, detector, evidence → prompt
models/InternVL3-2B/     pristine shared base weights (local copy, git-ignored)
adapters/                task LoRAs + deepfake heads (local copies, git-ignored)
ui/index.html            single-page UI (Chat · Anomaly detection · Finance)
scripts/                 live-model probes and dataset preparation
guides/                  integration guide and research report drafts
tests/                   pytest suite (runs without GPU or weights)
```

## Troubleshooting

- **"I couldn't load the … adapter"**: a path is wrong or the weights don't match the base. The server log has the full traceback. LoRA attach fails loudly when keys don't map ("Refusing silent base-model generation"). The service then drops the half-wrapped base, and the next request loads a clean one.
- **CUDA out of memory when switching to deepfake**: use the default `swap` policy, which keeps only one backbone resident.
- **Every chat turn routes to `general`**: the fine-tuned checkpoint is missing (`FINAL_LAYA_ROUTER_DIR`/`models/laya_router`) and receipts say `gate-unavailable`. Check the `routing` event's `reason`.
- **"The model is busy"**: another stream holds the lease. It is released when that stream finishes or its client disconnects.
