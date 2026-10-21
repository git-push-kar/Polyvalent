# Active adapters (external, pristine-base)

The previous in-repo adapters are quarantined at
`../quarantine/2026-10-21_fused_lr_and_ui/adapters/`. The active service loads
the adapters below from their training repositories so provenance stays with
the run that produced them. All of them (except the deepfake heads) are LoRA on
the same pristine base:

`C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`

| Routed task | Path | Format |
|---|---|---|
| `anomaly` | `C:\Users\Admin\Desktop\anomaly\outputs\tep_rca_internvl3_2b\internvl3_2b_pristine_rca` | PEFT LoRA r=16 / α=32 on q,k,v,o,gate,up,down |
| `reasoning` | `C:\Users\Admin\Desktop\C68-LR\models_v3\adapter\adapter` | PEFT LoRA r=64 / α=128 on q,k,v,o,gate,up,down |
| `deepfake` | `C:\Users\Admin\Desktop\LY_C68\deepfake_project\runs\intern_v2_ground\adapters` | `llm_adapter/` (r=64), `vision_adapter/` (r=16), `custom_heads/custom_heads.safetensors` |
| `general` | — | pristine base, no adapter |

`anomaly` and `reasoning` are attached to one shared `PeftModel` and
hot-swapped per request. `deepfake` runs in the dedicated
`DeepfakeReasoningModel` wrapper (its classifier heads have no InternVL
counterpart).

Override any path with `FINAL_BASE_DIR`, `FINAL_ANOMALY_DIR`,
`FINAL_REASONING_DIR`, `FINAL_DEEPFAKE_LLM_DIR`,
`FINAL_DEEPFAKE_VISION_DIR`, `FINAL_CUSTOM_HEADS`.
