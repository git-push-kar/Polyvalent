import sys, os, logging
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
logging.basicConfig(level=logging.WARNING)
import router._torch_shim  # noqa: F401
from router import AdapterService

svc = AdapterService(
    base_dir=r"C:\Users\Admin\Desktop\final\models\internvl3_2b_lr_fused",
    anomaly_dir=r"C:\Users\Admin\Desktop\final\adapters\Anomaly_adapter",
    deepfake_llm_dir=r"C:\Users\Admin\Desktop\final\adapters\Deepfake_adapter\llm_adapter_toplevel",
    deepfake_vision_dir=r"C:\Users\Admin\Desktop\final\adapters\Deepfake_adapter\vision_adapter_toplevel",
    custom_heads_path=r"C:\Users\Admin\Desktop\final\adapters\Deepfake_adapter\custom_heads\custom_heads.safetensors",
)
print(svc.select("general"), flush=True)
n = 0
for chunk in svc.stream("Say hello briefly.", max_new_tokens=6, stub=False):
    print(chunk.get("event"), repr(chunk.get("text", "")[:60]), flush=True)
    n += 1
print("CHUNKS:", n, flush=True)
