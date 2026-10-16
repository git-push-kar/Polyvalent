"""Load the ``tep_rca`` adapter onto a FRESH original InternVL3-2B / InternVL2-2B.

The adapter (adapter_config.json + adapter_model.safetensors) is independent:
it contains only the learned LoRA parameters. The base model is reloaded from
scratch and stays untouched except for the adapter being attached.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from llm.model import (
    assert_no_adapter,
    _drop_unused_inputs_embeds,
    ensure_language_model_generate,
)

logger = logging.getLogger(__name__)

ADAPTER_NAME = "tep_rca"


def resolve_adapter_dir(adapter_path: str, adapter_name: Optional[str] = None) -> Optional[Path]:
    """Return the directory containing ``adapter_config.json`` or None.

    Canonical training output (``Trainer.save_model`` on a PeftModel) is
    ``<adapter_dir>/<adapter_name>/adapter_config.json``. A legacy flat layout
    keeps ``<adapter_dir>/adapter_config.json``.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        return None
    if adapter_name:
        nested = adapter_path / adapter_name
        if (nested / "adapter_config.json").exists():
            return nested
    try:
        for sub in adapter_path.iterdir():
            if sub.is_dir() and (sub / "adapter_config.json").exists():
                return sub
    except OSError:
        pass
    if (adapter_path / "adapter_config.json").exists():
        return adapter_path
    return None


def load_tep_adapter(
    base_model: str,
    adapter_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Any = "auto",
    use_flash_attn: bool = False,
    trust_remote_code: bool = True,
    adapter_name: str = ADAPTER_NAME,
) -> Any:
    """Load a fresh base model (e.g. InternVL3-2B) and attach ONLY the tep_rca adapter.

    Args:
        base_model: path or hub id of the ORIGINAL base model (e.g. OpenGVLab/InternVL3-2B).
        adapter_path: directory produced by training (adapter_config.json etc).
        torch_dtype: inference dtype (default bfloat16, falling back to fp16).
        device_map: HF device_map for inference.

    Returns:
        The PeftModel (base + tep_rca adapter), ready for generate().
    """
    from peft import PeftModel
    from transformers import AutoModel

    adapter_path = Path(adapter_path)
    # Trainer.save_model on a PeftModel writes the adapter under
    # <adapter_name>/adapter_config.json; a plain save puts it at the root.
    # Prefer the nested (canonical) layout; fall back to the legacy flat copy.
    peft_dir = resolve_adapter_dir(adapter_path, adapter_name)
    if peft_dir is None:
        nested = adapter_path / adapter_name
        raise FileNotFoundError(
            f"No adapter_config.json in {adapter_path} (or {nested}); "
            f"is this a trained {adapter_name} adapter?"
        )
    dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float16)

    model = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        use_flash_attn=use_flash_attn,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    assert_no_adapter(model)
    ensure_language_model_generate(model)
    _drop_unused_inputs_embeds(model)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, use_fast=False
        )
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not set img_context_token_id: %s", exc)

    model = PeftModel.from_pretrained(
        model, str(peft_dir), adapter_name=adapter_name, is_trainable=False
    )
    logger.info("Attached adapter '%s' from %s onto base %s.",
                adapter_name, adapter_path, base_model)
    model.eval()
    return model