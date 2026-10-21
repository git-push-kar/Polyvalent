import sys
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
import router._torch_shim  # noqa: F401
from transformers import AutoModel

base = AutoModel.from_pretrained(
    r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B",
    trust_remote_code=True, torch_dtype="auto", device_map="cpu")
names = [n for n, _ in base.named_modules()
         if "layers.0.self_attn.q_proj" in n or "layers.0.mlp.down_proj" in n]
print("LIVE LLM PATHS:", names[:4])
vnames = [n for n, _ in base.named_modules() if "qkv" in n][:3]
print("LIVE VISION PATHS:", vnames)

from safetensors import safe_open
for label, p in [
    ("deepfake_llm",
     r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\runs"
     r"\intern_v2_ground\adapters\llm_adapter\adapter_model.safetensors"),
    ("anomaly",
     r"C:\Users\Admin\Desktop\anomaly\outputs\tep_rca_internvl3_2b"
     r"\internvl3_2b_pristine_rca\adapter_model.safetensors"),
]:
    keys = list(safe_open(p, framework="pt").keys())
    print(label, "n=", len(keys), "sample:", keys[0], "|", keys[1])
