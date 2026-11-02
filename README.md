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
request ──► │ Laya router (router/laya_router.py)          │
            │  pre_gate → laya_forward → arbitration       │
            │  → support guard (text cues / modality)      │
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

### Routing (Laya)

`router/laya_router.py` is the only router. It asks Laya one four-way question
(`deepfake / anomaly / reasoning / general`) and then applies a deterministic
**support guard**:

- An attached image always goes to `deepfake`. A 52-value sensor vector always goes to `anomaly`.
- Laya can only pick a specialist when the text contains cues for it. Otherwise the request falls back to `general`. For example, "what is your name" never reaches the deepfake adapter.
- "Use / load the reasoning adapter" forces `reasoning`. The directive is removed from the prompt before generation.
- A low-confidence abstention picks the specialist the cues point to, or `general` if there are none.

Every override is recorded as `reason=guard:*`. The raw Laya probabilities are
kept in `extra` and written to `logs/chat_runs.jsonl`.

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

A text-only "is this face fake?" request is answered with "attach an image"
**before** any loading starts, so the adapter currently loaded stays where it is.

---

## Requirements

- Python 3.10+ (developed on 3.11)
- `fastapi`, `uvicorn`, `pydantic>=2`, `numpy`
- For real generation: `torch`, `transformers`, `peft`, `safetensors`, `pillow`, `torchvision`
- `laya` (the router SDK). Without it, routing falls back to the support guard, and every receipt says `laya-unavailable`.
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

   For the Rieth et al. (2017) dataset, split the large files first:
   ```powershell
   python scripts/prepare_tep_testset.py --src TEP_FaultFree_Testing.csv TEP_Faulty_Testing.csv --out data/tep_test --runs 1
   ```
   If no dataset is found, the tab offers a few clearly labelled **synthetic** runs so the UI can still be tried. They are not TEP physics.
2. **Detection.** `POST /anomaly/run` replays the run through a detector:
   - With `FINAL_ANOMALY_REPO` set, it uses the trained `TEPApp` detector (LSTM-AE + CVA). That integration follows the API in the integration guide but has not been run against the real repo yet. If it fails, the service falls back to the built-in detector and the UI says so.
   - Otherwise it uses the built-in statistical detector: z-scores against a fault-free baseline, a rolling mean-square score, a threshold calibrated on normal data, and 6 consecutive alarms to confirm an event.
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
route        {"task": "reasoning", "label": ...}           Laya's choice
adapter      {"task", "loaded": [...], "unloaded": [...], "cold", "swap_ms"}
verdict      {"verdict": "REAL|FAKE", "confidence", "hotspots", "spectral"}   deepfake
thinking     {"text"}   deliberation (the UI shows it collapsed)
token        {"text"}   visible answer, streamed as it is generated
notice       {"text"}   UI feedback (attach an image, busy, load failure); not part of the conversation
done
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

## Repository layout

```
api/server.py            FastAPI app, SSE endpoints, lease handling
router/laya_router.py    Laya router + support guard
router/residency.py      lazy load / swap policy (torch-free, unit-tested)
router/adapter_service.py  real backends (PEFT attach/unload, deepfake wrapper), streaming
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
- **Every request routes to `general`**: `laya` is probably not installed. Check `extra.laya_available` in `/route/laya/predict` receipts.
- **"The model is busy"**: another stream holds the lease. It is released when that stream finishes or its client disconnects.
