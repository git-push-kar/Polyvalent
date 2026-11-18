# Hugging Face Release Manifest: PolyVL Models & Datasets

**Status:** **Successfully Uploaded & Live**  
**Account:** `Pushkarrrr15151`  
**Naming Standard:** `PolyVL`  

---

## 1. Live Hugging Face Repositories

| # | Repository Name | Type | Access | Licence | Live URL | Primary Contents |
|---|---|---|---|---|---|---|
| **1** | `Pushkarrrr15151/PolyVL-router-laya` | Model | Public | Apache-2.0 | [PolyVL-router-laya](https://huggingface.co/Pushkarrrr15151/PolyVL-router-laya) | Fine-tuned LAYA ModernBERT router head (`model.safetensors`, 808 MB, questions, tokenizer) |
| **2** | `Pushkarrrr15151/PolyVL-router-requests` | Dataset | Public | CC BY 4.0 | [PolyVL-router-requests](https://huggingface.co/datasets/Pushkarrrr15151/PolyVL-router-requests) | 2,500 reviewed rows (`train.jsonl`, `dev.jsonl`, `test.jsonl`), review decisions, data dictionary |
| **3** | `Pushkarrrr15151/PolyVL-tep-report-lora` | Model | Public | Apache-2.0 | [PolyVL-tep-report-lora](https://huggingface.co/Pushkarrrr15151/PolyVL-tep-report-lora) | TEP report-writer LoRA (`adapter_model.safetensors`, step 447, $r=16$, $\alpha=32$) |
| **4** | `Pushkarrrr15151/PolyVL-tep-report-distill` | Dataset | Public | CC BY 4.0 | [PolyVL-tep-report-distill](https://huggingface.co/datasets/Pushkarrrr15151/PolyVL-tep-report-distill) | TEP RCA distillation dataset (JSONL & Arrow splits, `split_metadata.json`, `teacher_manifest.json`) |
| **5** | `Pushkarrrr15151/PolyVL-logic-lora` | Model | Public | Apache-2.0 | [PolyVL-logic-lora](https://huggingface.co/Pushkarrrr15151/PolyVL-logic-lora) | Reasoning specialist LoRA ($v3\_sft$, `adapter_model.safetensors`, 295 MB, $r=64$, $\alpha=128$) |
| **6** | `Pushkarrrr15151/PolyVL-finance-rotation-lora` | Model | Public | Apache-2.0 | [PolyVL-finance-rotation-lora](https://huggingface.co/Pushkarrrr15151/PolyVL-finance-rotation-lora) | Finance LoRA adapter (`adapter_model.safetensors`, 16 MB) + numerical bridge (`ppo_heads.pt`, 9 MB), 49 tickers |
| **7** | `Pushkarrrr15151/PolyVL-deepfake-lora` | Model | **Gated** (`manual`) | Other / Gated | [PolyVL-deepfake-lora](https://huggingface.co/Pushkarrrr15151/PolyVL-deepfake-lora) | Dual LoRA (`llm_adapter`, `vision_adapter`), 5 custom classification heads, `loader.py`, `wrapper.py` |
| **8** | `Pushkarrrr15151/PolyVL-tep-protocol-a-lora` | Model (Secondary) | Public | Apache-2.0 | [PolyVL-tep-protocol-a-lora](https://huggingface.co/Pushkarrrr15151/PolyVL-tep-protocol-a-lora) | Stand-alone classifier comparison adapter (`adapter_model.safetensors`, step 300) |

---

## 2. Integrity and Sanitization Verification

All repositories have been sanitized and verified prior to upload:
- **Base Model Revision Pinning:** All InternVL3-2B adapters pinned to `OpenGVLab/InternVL3-2B` at revision commit `899155015275a9b7338c7f4677e19c784e0e5a21`.
- **Zero Personal Paths:** Absolute local machine paths completely removed.
- **Licence Safeguards:**
  - **Finance:** Proprietary raw exchange prices excluded; download script (`prepare_nifty50_data.py`) and 49 Nifty 50 ticker universe provided for reproduction.
  - **Deepfake:** Raw HydraFake face images and Veritas `sft_36k.json` data excluded; repository access set to **Gated (`manual`)**.
  - **TEP Distillation:** CC0 equivalent attribution provided for Rieth et al. 2017 (Harvard Dataverse).
- **Integrity:** `checksums.sha256` generated and committed to the root of every repository.
