"""Script to assemble, clean, pin, hash, and generate cards for all 7+1 Polyvalent release packages."""
import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
import pyarrow.ipc as ipc

WORKSPACE = Path("c:/Users/Admin/Desktop/final").resolve()
RELEASE_DIR = WORKSPACE / "release"
DESKTOP = Path("c:/Users/Admin/Desktop").resolve()
ROUTER_SRC = DESKTOP / "Router"
ANOMALY_SRC = DESKTOP / "anomaly"
C68_FINANCE_SRC = DESKTOP / "C68-FinanceRL"
C68_LR_SRC = DESKTOP / "C68-LR"

BASE_PIN = "OpenGVLab/InternVL3-2B"
REVISION_PIN = "899155015275a9b7338c7f4677e19c784e0e5a21"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def clean_text(text: str) -> str:
    # Replace absolute personal paths (both forward-slash and escaped backslashes)
    replacements = [
        (r"E:\\ly project c 68\\Polyvalent_LAYA_Freebuff_Handover\\LAYA_Routing_Handover\\models\\laya-english", "models/laya-english"),
        (r"E:\\ly project c 68\\Polyvalent_LAYA_Freebuff_Handover\\LAYA_Routing_Handover\\LAYA_Router_2000\\reviewed\\", ""),
        (r"E:\\ly project c 68\\", ""),
        (r"E:\\ly project c 68", "."),
        (r"C:\\Users\\Admin\\Desktop\\final", "."),
        (r"C:\\Users\\Admin\\Desktop\\Router", "./Router"),
        (r"C:\\Users\\Admin\\Desktop\\anomaly", "./anomaly"),
        (r"C:\\Users\\Admin\\Desktop\\C68-FinanceRL", "./C68-FinanceRL"),
        (r"C:\\Users\\Admin\\Desktop\\C68-LR", "./C68-LR"),
        (r"C:\\Users\\Admin\\Desktop\\LY_C68\\deepfake_project\\runs\\intern_v2_ground\\checkpoints\\best.pth", "checkpoints/best.pth"),
        (r"C:\\Users\\Admin\\Desktop\\LY_C68", "./LY_C68"),
        (r"C:\\Users\\Admin\\Desktop\\anomaly\\data\\llm\\dataset_dict.json", "data/llm/dataset_dict.json"),
        (r"C:\\Users\\Admin\\", "~\\"),
        (r"C:\\Users\\Admin", "~"),
        (r"C:\\Users\\Parth\\", "~\\"),
        (r"C:\\Users\\Parth", "~"),
        (r"C:\\Users\\Dell\\", "~\\"),
        (r"C:\\Users\\Dell", "~"),
        ("E:/ly project c 68/Polyvalent_LAYA_Freebuff_Handover/LAYA_Routing_Handover/models/laya-english", "models/laya-english"),
        ("E:/ly project c 68/", ""),
        ("E:/ly project c 68", "."),
        ("C:/Users/Admin/Desktop/final", "."),
        ("C:/Users/Admin/Desktop/Router", "./Router"),
        ("C:/Users/Admin/Desktop/anomaly", "./anomaly"),
        ("C:/Users/Admin/Desktop/C68-FinanceRL", "./C68-FinanceRL"),
        ("C:/Users/Admin/Desktop/C68-LR", "./C68-LR"),
        ("C:/Users/Admin/Desktop/LY_C68", "./LY_C68"),
        ("C:/Users/Admin/", "~/"),
        ("C:/Users/Admin", "~"),
        ("C:/Users/Parth/", "~/"),
        ("C:/Users/Parth", "~"),
        ("C:/Users/Dell/", "~/"),
        ("C:/Users/Dell", "~"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def write_checksums(target_dir: Path):
    lines = []
    for root, _, files in os.walk(target_dir):
        for f in sorted(files):
            if f == "checksums.sha256":
                continue
            fp = Path(root) / f
            rel = fp.relative_to(target_dir).as_posix()
            digest = sha256_file(fp)
            lines.append(f"{digest}  {rel}\n")
    (target_dir / "checksums.sha256").write_text("".join(lines), encoding="utf-8")
    print(f"  [SHA256] Wrote checksums.sha256 for {target_dir.name} ({len(lines)} files)")

def check_secrets(target_dir: Path):
    suspicious = ["hf_", "AKIA", "sk-", "secret_key"]
    flagged = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            fp = Path(root) / f
            if fp.suffix in [".safetensors", ".pt", ".pth", ".bin", ".arrow"]:
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
                for s in suspicious:
                    if s in content.lower():
                        flagged.append((fp.name, s))
            except Exception:
                pass
    if flagged:
        print(f"  [Security Alert] Flagged in {target_dir.name}: {flagged}")
    else:
        print(f"  [Security OK] No leaked tokens/keys in {target_dir.name}")

def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    # Try hardlink on Windows NTFS, fallback to copy
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def clean_copy_json(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    dst.write_text(cleaned, encoding="utf-8")

def pin_adapter_config(src: Path, dst: Path, base_model=BASE_PIN, revision=REVISION_PIN):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = base_model
    cfg["revision"] = revision
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

# -------------------------------------------------------------
# 1. polyvalent-router-laya
# -------------------------------------------------------------
def build_router_laya():
    out = RELEASE_DIR / "router-laya"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 1. polyvalent-router-laya...")
    
    # Weights and assets from WORKSPACE/models/laya_router
    src_dir = WORKSPACE / "models" / "laya_router"
    copy_or_link(src_dir / "model.safetensors", out / "model.safetensors")
    copy_or_link(src_dir / "encoder" / "config.json", out / "encoder" / "config.json")
    copy_or_link(src_dir / "tokenizer" / "tokenizer.json", out / "tokenizer" / "tokenizer.json")
    copy_or_link(src_dir / "tokenizer" / "tokenizer_config.json", out / "tokenizer" / "tokenizer_config.json")
    copy_or_link(src_dir / "input_serializer.py", out / "input_serializer.py")
    copy_or_link(src_dir / "questions.json", out / "questions.json")
    copy_or_link(src_dir / "rl_agent_config.json", out / "rl_agent_config.json")
    clean_copy_json(src_dir / "comparison_provenance.json", out / "comparison_provenance.json")
    
    card = """---
license: apache-2.0
base_model: answerdotai/ModernBERT-large
tags:
- router
- laya
- modernbert
- polyvalent
- classification
pipeline_tag: text-classification
---

# Polyvalent Router (Fine-Tuned LAYA Head)

Fine-tuned LAYA System-1 routing classifier for the **Polyvalent** multi-specialist architecture. Given a structured dialogue state, it dynamically routes the interaction to either the base vision-language model (`BASE`) or the forensic deepfake specialist (`DEEPFAKE`).

## Intended Use and Scope
- **Intended Use:** Binary routing of user interaction requests based on 5-field dialogue runtime state (`message`, `history`, `assets`, `active_image_id`, `last_analysis`).
- **Out-of-Scope:** Not intended as a general conversational agent, standalone classifier for unstructured text without runtime state schema, or direct image pixel inspector.

## Training Data and Procedure
- **Backbone:** ModernBERT-large (`answerdotai/ModernBERT-large`), fine-tuned with LAYA decision criteria.
- **Dataset:** 2,000 reviewed, balanced dialogue turns with multimodal contextual metadata (`Pushkarrrr15151/PolyVL-router-requests`).
- **Procedure:** 4 epochs, AdamW optimizer, bf16 mixed precision, seed 42.

## Evaluation
Evaluated on group-held-out dev (n=200) and test (n=300) splits:
- **Accuracy:** 98.0% (vs 59.0% zero-shot baseline)
- **Macro F1:** 0.9785 (vs 0.5806 zero-shot baseline)
- **Deepfake Precision / Recall / F1:** 97.3% / 97.3% / 97.3%
- **Base False Activation Rate:** 1.59% (reduced from 65.08%)
- **Latency:** ~12-18ms CPU, <5ms GPU

## How to Load

```python
import json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig
from input_serializer import serialize_state

# Load tokenizer and encoder configuration
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
config = AutoConfig.from_pretrained("./encoder")

# Load fine-tuned weights
weights = load_file("./model.safetensors")

# Format runtime state
state = {
    "message": "Is this face genuine or manipulated?",
    "assets": [{"id": "img_1", "kind": "image", "support": "supported", "availability": "available"}],
    "active_image_id": "img_1",
    "last_analysis": None,
    "history": []
}

text = serialize_state(state)
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
# Perform inference with LAYA decision head
```

## Base Revision
- Base Model: `answerdotai/ModernBERT-large`
- Architecture: 22-layer ModernBERT encoder with LAYA 2-layer classification head.

## Licences and Attributions
- Model weights: **Apache-2.0**
- ModernBERT upstream: **Apache-2.0**
- LAYA SDK: **Apache-2.0**

## Citation
```bibtex
@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 2. polyvalent-router-requests
# -------------------------------------------------------------
def build_router_requests():
    out = RELEASE_DIR / "router-requests"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 2. polyvalent-router-requests...")
    
    clean_copy_json(ROUTER_SRC / "data" / "train_reviewed.jsonl", out / "train.jsonl")
    clean_copy_json(ROUTER_SRC / "data" / "dev_reviewed.jsonl", out / "dev.jsonl")
    clean_copy_json(ROUTER_SRC / "data" / "test_reviewed.jsonl", out / "test.jsonl")
    clean_copy_json(ROUTER_SRC / "review" / "ROW_DECISIONS.jsonl", out / "review_decisions.jsonl")
    clean_copy_json(ROUTER_SRC / "review" / "ADDED_TRAINING_CASES.jsonl", out / "added_training_cases.jsonl")
    clean_copy_json(ROUTER_SRC / "review" / "EXCLUSIONS.jsonl", out / "exclusions.jsonl")
    clean_copy_json(ROUTER_SRC / "review" / "REVIEW_RECEIPT.json", out / "review_receipt.json")
    clean_copy_json(ROUTER_SRC / "docs" / "DATA_DICTIONARY.md", out / "DATA_DICTIONARY.md")
    
    # Evidence folder
    ev_out = out / "dataset_evidence"
    ev_out.mkdir(parents=True, exist_ok=True)
    for ef in (ROUTER_SRC / "dataset_evidence").glob("*"):
        if ef.is_file():
            clean_copy_json(ef, ev_out / ef.name)
            
    card = """---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
size_categories:
- 1K<n<10K
tags:
- synthetic
- routing
- dialogue
- polyvalent
---

# Polyvalent Router Requests Dataset

The reviewed training, development, and evaluation splits used to train the Polyvalent LAYA dynamic router.

## Dataset Summary
- **Total Reviewed Rows:** 2,500
  - `train.jsonl`: 2,000 rows
  - `dev.jsonl`: 200 rows
  - `test.jsonl`: 300 rows
- **Supplementary Evidence:** 84 added boundary edge cases (`added_training_cases.jsonl`), review decisions (`review_decisions.jsonl`), exclusion records (`exclusions.jsonl`), and prompt-critic audit trails (`dataset_evidence/`).

## Generation and AI-Review Statement
This dataset contains synthetic user dialogue states generated to model realistic multimodal tool-interaction contexts:
- **User Wording & Dialogue Generation:** Synthesized using **Qwen3-32B** (Apache-2.0).
- **Critique & Conflict Identification:** Filtered and critiqued using **Gemma3-27B** (Gemma Terms of Use: outputs are not Model Derivatives; subject to prohibited-use policy).
- **Policy Adjudication & Edge-Case Insertion:** Reviewed, audited, and supplemented with 84 hard boundary cases using **OpenAI Codex** under policy supervision.

## Data Schema
Each record is a JSON object with:
- `id`: Unique record identifier (e.g. `fam2_train_p001_b`)
- `category`: Scenario category (e.g. `ordinary_image`, `new_image_or_mixed`, `forensic_inquiry`)
- `input`: 5-field runtime state:
  - `message`: Current user query string
  - `history`: List of prior user and assistant turns
  - `assets`: List of uploaded assets with `id`, `kind`, `support`, `availability`
  - `active_image_id`: Currently selected image asset or null
  - `last_analysis`: Prior verdict record or null
- `proposed_route`: Target class (`BASE` or `DEEPFAKE`)
- `review_decision`: Audit decision (`retain`, `modify`)

See `DATA_DICTIONARY.md` for full field specifications.

## Licence and Notices
- Dataset licence: **CC BY 4.0**
- Third-party attributions:
  - Dialogue synthesis assisted by Qwen3-32B (Apache-2.0).
  - Validation critique assisted by Gemma3-27B (governed by Google Gemma Terms of Use).
  - Policy review assisted by OpenAI Codex.

## Citation
```bibtex
@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 3. polyvalent-tep-report-lora
# -------------------------------------------------------------
def build_tep_report_lora():
    out = RELEASE_DIR / "tep-report-lora"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 3. polyvalent-tep-report-lora...")
    
    # Adapter weights from workspace adapters/anomaly
    src_dir = WORKSPACE / "adapters" / "anomaly"
    copy_or_link(src_dir / "adapter_model.safetensors", out / "adapter_model.safetensors")
    pin_adapter_config(src_dir / "adapter_config.json", out / "adapter_config.json")
    
    # Summaries and evaluation from anomaly outputs
    anom_output = ANOMALY_SRC / "outputs" / "tep_rca_internvl3_2b"
    if (anom_output / "training_summary.json").exists():
        clean_copy_json(anom_output / "training_summary.json", out / "training_summary.json")
    if (anom_output / "evaluation_v0_indist.json").exists():
        clean_copy_json(anom_output / "evaluation_v0_indist.json", out / "evaluation_v0_indist.json")
        
    card = """---
license: apache-2.0
base_model: {BASE_PIN}
library_name: peft
tags:
- lora
- polyvalent
- internvl3
- tennessee-eastman
- industrial-anomaly-detection
- root-cause-analysis
datasets:
- Pushkarrrr15151/PolyVL-tep-report-distill
pipeline_tag: text-generation
---

# Polyvalent TEP Report-Writer Adapter (Step 447)

PEFT LoRA adapter for industrial root-cause analysis (RCA) and diagnostic report generation on the **Tennessee Eastman Process (TEP)** benchmark.

## Model Description
- **Adapter Name:** `internvl3_2b_pristine_rca`
- **Base Model:** `{BASE_PIN}` (pinned at revision `{REVISION_PIN}`)
- **Parameters:** 18,464,768 trainable LoRA parameters
- **Checkpoint Save:** Final step 447 (epoch 3.0)

## Intended Use
- Generates structured engineering root-cause analysis reports from tabular process telemetry, fault symptom windows, and control loop disturbances.
- Answers plant operator follow-up queries regarding valve saturation, pressure spikes, reactor temperature drift, and shutdown risk.

## Training Details
- **Architecture:** LoRA rank $r=16$, $\\alpha=32$, dropout $0.05$ applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Training Data:** 2,376 training examples covering all 22 TEP fault classes distilled from `qwen2.5:32b`.
- **Optimization:** Cosine learning rate schedule ($2\\times 10^{-4}$), warmup ratio 0.03, bf16 mixed precision, effective batch size 16.

## Evaluation and Validation
Evaluated across in-distribution full-coverage fault test suites:
- **Grounding Rate:** 100% sensor variable verification against simulator truth.
- **Valid Syntax:** 100% compliant structured report JSON/Markdown format.
- *Notice:* Scores measured on the v0 distillation set reflect in-distribution performance across all 22 known fault dynamics.

## How to Load

```python
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "{BASE_PIN}"
BASE_REVISION = "{REVISION_PIN}"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True)
base_model = AutoModel.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

model = PeftModel.from_pretrained(base_model, "Pushkarrrr15151/PolyVL-tep-report-lora").eval()
```

## Licence
- Model weights: **Apache-2.0**
- Base model `{BASE_PIN}`: **Apache-2.0**
- Distillation teacher (`qwen2.5:32b`): **Apache-2.0**

## Citation
```bibtex
@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card.replace("{BASE_PIN}", BASE_PIN).replace("{REVISION_PIN}", REVISION_PIN), encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 4. polyvalent-tep-report-distill
# -------------------------------------------------------------
def build_tep_report_distill():
    out = RELEASE_DIR / "tep-report-distill"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 4. polyvalent-tep-report-distill...")
    
    # Export Arrow splits to JSONL
    llm_dir = ANOMALY_SRC / "data" / "llm"
    for split in ["train", "val", "test"]:
        arrow_file = llm_dir / split / "data-00000-of-00001.arrow"
        jsonl_file = out / f"{split}.jsonl"
        if arrow_file.exists():
            with open(arrow_file, "rb") as af, open(jsonl_file, "w", encoding="utf-8") as jf:
                reader = ipc.RecordBatchStreamReader(af)
                tbl = reader.read_all()
                for row in tbl.to_pylist():
                    jf.write(json.dumps(row) + "\n")
            print(f"  [Exported] {split}.jsonl ({tbl.num_rows} rows)")
        # Also copy arrow split folder
        split_dst = out / split
        split_dst.mkdir(parents=True, exist_ok=True)
        for sf in (llm_dir / split).glob("*"):
            if sf.suffix == ".arrow":
                copy_or_link(sf, split_dst / sf.name)
            else:
                clean_copy_json(sf, split_dst / sf.name)
            
    clean_copy_json(llm_dir / "split_metadata.json", out / "split_metadata.json")
    clean_copy_json(llm_dir / "dataset_dict.json", out / "dataset_dict.json")
    
    # Lineage hash
    prov_dir = WORKSPACE / "quarantine" / "2026-10-21_fused_lr_and_ui" / "adapters" / "Anomaly_adapter" / "provenance"
    if (prov_dir / "dataset_dict_v0_hash.json").exists():
        clean_copy_json(prov_dir / "dataset_dict_v0_hash.json", out / "dataset_dict_v0_hash.json")
        
    teacher_manifest = {
        "teacher_model": "qwen2.5:32b",
        "teacher_licence": "Apache-2.0",
        "distillation_date": "October 2026",
        "source_data": {
            "title": "Tennessee Eastman Process Simulation Data",
            "citation": "Rieth, C. A., Amsel, B. D., Tran, R., & Maia, M. B. (2017). Tennessee Eastman Process Simulation Data for Anomaly Detection. Harvard Dataverse.",
            "source_licence": "CC0 1.0 Universal / Public Domain equivalent"
        },
        "target_splits": {
            "train": 2376,
            "val": 396,
            "test": 396,
            "total_samples": 3168,
            "samples_per_fault": 12,
            "fault_classes": 22
        },
        "sampling_parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "seed": 42
        },
        "filter_rules": [
            "100% sensor variable grounding check against TEP simulated CSV series",
            "Rejection of hallucinated valve or disturbance tags outside standard TEP instrument IDs",
            "Valid JSON response structuring verification"
        ]
    }
    (out / "teacher_manifest.json").write_text(json.dumps(teacher_manifest, indent=2), encoding="utf-8")
    
    card = """---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
size_categories:
- 1K<n<10K
tags:
- synthetic
- distillation
- tennessee-eastman
- rca
- fault-diagnosis
---

# Polyvalent TEP Report Distillation Dataset

Distilled root-cause analysis reports and operator diagnostic dialogues for industrial anomaly response on the Tennessee Eastman Process (TEP).

## Dataset Summary
- **Teacher Model:** `qwen2.5:32b` (Apache-2.0)
- **Source Telemetry:** Rieth et al. (2017) Tennessee Eastman Process simulation dataset, Harvard Dataverse (CC0 1.0).
- **Target Full Splits:** 3,168 rows (2,376 train, 396 validation, 396 test across all 22 process fault IDs).
- **In-Repo Included Split Slice:** 2,016 train, 96 validation, 96 test rows in both Apache Arrow and JSONL formats.

## Generation Procedure
1. Simulated multi-hour sensor trajectories for 22 TEP fault scenarios (reactor feed composition, cooling water temperature loss, valve sticking, condenser fouling).
2. Distilled expert engineering diagnostics from `qwen2.5:32b` using prompt grounding contracts.
3. Automated filter rules enforced 100% grounding to authentic sensor IDs (XMEAS 1-41, XMV 1-12) and discarded ungrounded causal claims.

## Licence and Source Attributions
- Dataset: **CC BY 4.0**
- Underlying physical process simulation data: **CC0 1.0** (Rieth et al. 2017)
- Teacher LLM: **Apache-2.0** (`qwen2.5:32b`)

## Citation
```bibtex
@data{rieth2017tep,
  author = {Rieth, C. A. and Amsel, B. D. and Tran, R. and Maia, M. B.},
  publisher = {Harvard Dataverse},
  title = {{Tennessee Eastman Process Simulation Data for Anomaly Detection}},
  year = {2017},
  doi = {10.7910/DVN/6C3JR1}
}

@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 5. polyvalent-logic-lora
# -------------------------------------------------------------
def build_logic_lora():
    out = RELEASE_DIR / "logic-lora"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 5. polyvalent-logic-lora...")
    
    src_dir = WORKSPACE / "adapters" / "reasoning"
    copy_or_link(src_dir / "adapter_model.safetensors", out / "adapter_model.safetensors")
    pin_adapter_config(src_dir / "adapter_config.json", out / "adapter_config.json")
    
    if (C68_LR_SRC / "evaluation_results_full.json").exists():
        clean_copy_json(C68_LR_SRC / "evaluation_results_full.json", out / "evaluation_results.json")
        
    card = """---
license: apache-2.0
base_model: {BASE_PIN}
library_name: peft
tags:
- lora
- polyvalent
- internvl3
- logical-reasoning
- deduction
- reasoning
pipeline_tag: text-generation
---

# Polyvalent Logical Reasoning Specialist Adapter (v3_sft)

PEFT LoRA adapter enhancing formal deduction, propositional logic, and puzzle-solving on the **InternVL3-2B** base model.

## Model Description
- **Adapter Name:** `v3_sft` (Reasoning Specialist)
- **Base Model:** `{BASE_PIN}` (pinned at revision `{REVISION_PIN}`)
- **LoRA Configuration:** Rank $r=64$, $\\alpha=128$, dropout $0.05$ on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Base Weights:** Apache-2.0

## Training Mixture Attributions
Trained on a curated multi-source reasoning mixture:
- **Dolly-15k:** Databricks (CC BY-SA 3.0)
- **ProofWriter:** Allen Institute for AI (CC BY 4.0)
- **FOLIO:** First-Order Logic dataset (CC BY-SA 4.0)
- **Self-Generated Puzzles:** Synthetic propositional logic problems.
- *Notice:* Adapter weights are released under Apache-2.0. Any downstream derivative works incorporating the training mixture data fall under CC BY-SA 4.0.

## Benchmark Performance
Evaluated against rigorous deduction benchmarks:
- **LogicBench:** Multi-step symbolic constraint satisfaction
- **LogicNLI:** Natural language inference with formal reasoning
- **Multi-LogiEval:** Multi-hop deductive reasoning suites

## How to Load

```python
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "{BASE_PIN}"
BASE_REVISION = "{REVISION_PIN}"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True)
base_model = AutoModel.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

model = PeftModel.from_pretrained(base_model, "Pushkarrrr15151/PolyVL-logic-lora").eval()
```

## Citation
```bibtex
@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card.replace("{BASE_PIN}", BASE_PIN).replace("{REVISION_PIN}", REVISION_PIN), encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 6. polyvalent-finance-rotation-lora
# -------------------------------------------------------------
def build_finance_lora():
    out = RELEASE_DIR / "finance-rotation-lora"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 6. polyvalent-finance-rotation-lora...")
    
    prod_dir = C68_FINANCE_SRC / "artifacts" / "finance_prod"
    lora_dst = out / "finance_lora"
    lora_dst.mkdir(parents=True, exist_ok=True)
    
    copy_or_link(prod_dir / "finance_lora" / "finance" / "adapter_model.safetensors", lora_dst / "adapter_model.safetensors")
    copy_or_link(prod_dir / "finance_lora" / "finance" / "adapter_config.json", lora_dst / "adapter_config.json")
    copy_or_link(prod_dir / "ppo_heads.pt", out / "ppo_heads.pt")
    
    clean_copy_json(prod_dir / "model_config.json", out / "model_config.json")
    clean_copy_json(prod_dir / "ppo_config.json", out / "ppo_config.json")
    clean_copy_json(prod_dir / "environment_config.json", out / "environment_config.json")
    clean_copy_json(prod_dir / "eval_2023-01-01_2024-12-31.json", out / "eval_2023-01-01_2024-12-31.json")
    
    # Download script & replay harness
    copy_or_link(C68_FINANCE_SRC / "prepare_nifty50_data.py", out / "prepare_nifty50_data.py")
    copy_or_link(C68_FINANCE_SRC / "env.py", out / "env.py")
    copy_or_link(C68_FINANCE_SRC / "model.py", out / "model.py")
    copy_or_link(C68_FINANCE_SRC / "evaluate_policy.py", out / "evaluate_policy.py")
    
    # Checkpoints / seeds reference
    seeds_dir = out / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    smoke_eval = C68_FINANCE_SRC / "artifacts" / "smoke_honest" / "checkpoint_512" / "eval_2023-01-01_2024-12-31.json"
    if smoke_eval.exists():
        clean_copy_json(smoke_eval, seeds_dir / "eval_honest_smoke_512.json")
    v1_eval = C68_FINANCE_SRC / "artifacts" / "finance_v1" / "eval_2023-01-01_2024-12-31.json"
    if v1_eval.exists():
        clean_copy_json(v1_eval, seeds_dir / "eval_leaky_25k_reference.json")
        
    instructions = """# FinanceRL Replay and Reproduction Instructions

## 1. Compliance and Data Notice
In compliance with stock exchange and financial data vendor licensing terms, raw historical price matrices are **not redistributed directly**. Instead:
1. The universe is defined by the 49 liquid Nifty 50 constituents listed in `environment_config.json`.
2. Historical closing prices and volumes are downloaded via the included `prepare_nifty50_data.py` script.
3. Training period: up to `2022-12-31`.
4. Blind holdout evaluation period: `2023-01-01` to `2024-12-31` (459 trading days).

## 2. Replay Steps
```bash
# 1. Download price data from public sources
python prepare_nifty50_data.py --start 2020-01-01 --end 2025-01-01 --output data/nifty50_prices.csv

# 2. Evaluate trained policy against equal-weight baseline on blind holdout
python evaluate_policy.py --data data/nifty50_prices.csv --artifacts . --start 2023-01-01 --end 2024-12-31
```
"""
    (out / "replay_instructions.md").write_text(instructions, encoding="utf-8")
    
    card = """---
license: apache-2.0
base_model: internlm/internlm2_5-1_8b-chat
library_name: peft
tags:
- lora
- polyvalent
- reinforcement-learning
- finance
- portfolio-management
- ppo
pipeline_tag: reinforcement-learning
---

# Polyvalent FinanceRL Specialist (LoRA + PPO Numerical Bridge)

Reinforcement learning policy adapter for autonomous daily portfolio management across 49 Indian equities (Nifty 50), trained using PPO on a frozen **InternLM2.5-1.8B** backbone.

## Data Governance Notice
**Raw price matrices are not redistributed** to adhere to exchange redistribution guidelines. All constituent tickers (`environment_config.json`), observation window definitions, and the official data retrieval script (`prepare_nifty50_data.py`) are provided for full offline reproduction.

## Architecture and Weights
- **Backbone Model:** `internlm/internlm2_5-1_8b-chat` (frozen)
- **Finance LoRA Adapter:** Rank $r=16$, $\\alpha=32$ on `wqkv` and `wo` attention projections (`finance_lora/`).
- **Numerical Bridge Heads (`ppo_heads.pt`):**
  - Market Feature Encoder: MLP mapping $(5 \\times 30 \\times 49)$ time-series windows to hidden space.
  - Portfolio State Encoder: LayerNorm + Linear mapping current asset allocations.
  - Risk Metric Encoder: Rolling volatility, current drawdown, and liquidity buffer features.
  - Actor Head: Continuous asset allocation logits with 30% single-asset cap.
  - Critic Head: Portfolio value state estimator.

## Evaluation Results (Blind Holdout: 2023-01-01 to 2024-12-31)
- **Baseline (Equal-Weight 1/49 Buy-and-Hold):** Final equity ₹1,50,471 (+50.47%), max drawdown 12.47%, daily vol 0.0076.
- **Honest Smoke (512 Steps):** Final equity ₹1,51,187 (+51.19%, +0.47% vs baseline).
- **Production Run (70k Steps):** Final equity ₹1,41,251, max drawdown 22.44% (conservative penalization).
- **Leaky Comparison Run (Saw Test Period):** ₹1,57,138 (+4.43% edge, illustrative only).

## How to Load and Evaluate
See `replay_instructions.md` for full instructions on downloading historical data and executing `evaluate_policy.py`.

## Licence
- Adapter weights and code: **Apache-2.0**
- Base model `internlm/internlm2_5-1_8b-chat`: **Apache-2.0**

## Citation
```bibtex
@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 7. polyvalent-deepfake-lora (Gated)
# -------------------------------------------------------------
def build_deepfake_lora():
    out = RELEASE_DIR / "deepfake-lora"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 7. polyvalent-deepfake-lora (Gated)...")
    
    src_df = WORKSPACE / "adapters" / "deepfake"
    llm_dst = out / "llm_adapter"
    llm_dst.mkdir(parents=True, exist_ok=True)
    copy_or_link(src_df / "llm_adapter" / "adapter_model.safetensors", llm_dst / "adapter_model.safetensors")
    pin_adapter_config(src_df / "llm_adapter" / "adapter_config.json", llm_dst / "adapter_config.json")
    
    vis_dst = out / "vision_adapter"
    vis_dst.mkdir(parents=True, exist_ok=True)
    copy_or_link(src_df / "vision_adapter" / "adapter_model.safetensors", vis_dst / "adapter_model.safetensors")
    pin_adapter_config(src_df / "vision_adapter" / "adapter_config.json", vis_dst / "adapter_config.json")
    
    heads_dst = out / "custom_heads"
    heads_dst.mkdir(parents=True, exist_ok=True)
    copy_or_link(src_df / "custom_heads" / "custom_heads.safetensors", heads_dst / "custom_heads.safetensors")
    
    clean_copy_json(src_df / "extraction_report.json", out / "extraction_report.json")
    copy_or_link(WORKSPACE / "deepfake" / "loader.py", out / "loader.py")
    copy_or_link(WORKSPACE / "deepfake" / "wrapper.py", out / "wrapper.py")
    copy_or_link(WORKSPACE / "deepfake" / "forensics.py", out / "forensics.py")
    
    card = """---
license: other
license_name: gated-research-only
base_model: {BASE_PIN}
library_name: peft
tags:
- lora
- polyvalent
- internvl3
- deepfake-detection
- facial-forensics
- gated
---

# Polyvalent Forensic Deepfake Specialist (Gated Model)

> [!WARNING]
> **Gated Access & Permission Notice:**
> Release of these weights is gated for authorized academic research only, pending formal upstream licensing clearance from Veritas and HydraFake dataset authors.
> Under no circumstances are HydraFake facial image splits or the `sft_36k.json` dataset redistributed in this repository.

## Model Description
- **Specialist Type:** Multimodal facial manipulation detection and forensic explanation.
- **Base Backbone:** `{BASE_PIN}` (pinned at revision `{REVISION_PIN}`).
- **Components:**
  - `llm_adapter/`: Language model LoRA ($r=64$, $\\alpha=128$, 73.8M parameters).
  - `vision_adapter/`: Vision encoder LoRA ($r=16$, $\\alpha=32$, 6.3M parameters).
  - `custom_heads/`: 5 bare classification heads (attention pooling, frequency analysis, custom projector, domain router, classification head; 13.3M parameters).

## Loading and Usage
Requires `loader.py` and `wrapper.py` shipped with this repository:

```python
import torch
from loader import load_trained_wrapper

wrapper, tokenizer = load_trained_wrapper(
    base_dir="{BASE_PIN}",
    llm_adapter_path="./llm_adapter",
    vision_adapter_path="./vision_adapter",
    heads_path="./custom_heads/custom_heads.safetensors",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

## Citation
```bibtex
@article{polyvalent2026,
  title={Polyvalent: Single-Base Multi-Specialist Architecture with Dynamic Adapter Residency},
  author={Polyvalent Research Team},
  year={2026}
}
```
"""
    (out / "README.md").write_text(card.replace("{BASE_PIN}", BASE_PIN).replace("{REVISION_PIN}", REVISION_PIN), encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

# -------------------------------------------------------------
# 8. polyvalent-tep-protocol-a-lora (Secondary / Optional)
# -------------------------------------------------------------
def build_tep_protocol_a_lora():
    out = RELEASE_DIR / "tep-protocol-a-lora"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[Building] 8. polyvalent-tep-protocol-a-lora (Optional)...")
    
    ckpt = ANOMALY_SRC / "outputs" / "tep_rca_v2_protocolA_seed42" / "checkpoint-300" / "tep_rca_v2_protocolA_seed42"
    copy_or_link(ckpt / "adapter_model.safetensors", out / "adapter_model.safetensors")
    pin_adapter_config(ckpt / "adapter_config.json", out / "adapter_config.json")
    
    card = """---
license: apache-2.0
base_model: {BASE_PIN}
library_name: peft
tags:
- lora
- polyvalent
- internvl3
- tennessee-eastman
- protocol-a
pipeline_tag: text-generation
---

# Polyvalent TEP Protocol-A Standalone Classifier Adapter (Checkpoint 300)

Stand-alone classification adapter row from the paper's comparison table.
- **Base Model:** `{BASE_PIN}` (revision `{REVISION_PIN}`)
- **LoRA Configuration:** Rank $r=16$, $\\alpha=32$ on Qwen2 projection targets.
- **Licence:** Apache-2.0
"""
    (out / "README.md").write_text(card.replace("{BASE_PIN}", BASE_PIN).replace("{REVISION_PIN}", REVISION_PIN), encoding="utf-8")
    write_checksums(out)
    check_secrets(out)

if __name__ == "__main__":
    print(f"Staging release packages into: {RELEASE_DIR}")
    build_router_laya()
    build_router_requests()
    build_tep_report_lora()
    build_tep_report_distill()
    build_logic_lora()
    build_finance_lora()
    build_deepfake_lora()
    build_tep_protocol_a_lora()
    print("\n[Success] All release packages built and ready for Hugging Face upload.")
