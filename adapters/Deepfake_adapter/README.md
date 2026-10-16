# Deepfake adapter layout (final/)

- `llm_adapter/` + `vision_adapter/` — ORIGINALS as extracted by
  `deepfake_project/extract_adapters.py`. Key namespace is submodule-relative
  (`base_model.model.model.layers...`, `base_model.model.encoder.layers...`)
  because that harness wrapped `backbone.language_model` / `backbone.vision_model`
  directly. Do not delete (provenance).
- `llm_adapter_toplevel/` + `vision_adapter_toplevel/` — mechanical re-key
  copies (script `scripts/rekey_deepfake_adapters.py`, 392+192 keys, 0 untouched):
  `...model.model.layers...` -> `...model.language_model.model.layers...`,
  `...model.encoder.layers...` -> `...model.vision_model.encoder.layers...`,
  so they attach to raw InternVL3-2B in the single shared PeftModel.
  These are what `router/adapter_service.py` loads.
- `custom_heads/custom_heads.safetensors` (59 bare-key tensors) — classifier /
  pooling / FFT / projector / domain-router weights for the
  `DeepfakeReasoningModel` wrapper (`LY_C68/deepfake_project/model.py`).
  They match NO module on raw InternVL and are intentionally NOT loaded until
  the wrapper is ported (service asserts backbone-key disjointness first).
  Status: deterministic verdict/hotspot branch pending wrapper port;
  LLM reasoning stream already uses the re-keyed forensic LoRA.
