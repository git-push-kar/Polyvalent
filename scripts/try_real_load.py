import sys, os, time
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)
import router._torch_shim  # noqa: F401
from router import AdapterService

svc = AdapterService(
    base_dir=r"C:\Users\Admin\Desktop\final\models\internvl3_2b_lr_fused",
    anomaly_dir=r"C:\Users\Admin\Desktop\final\adapters\Anomaly_adapter",
    deepfake_llm_dir=r"C:\Users\Admin\Desktop\final\adapters\Deepfake_adapter\llm_adapter_toplevel",
    deepfake_vision_dir=r"C:\Users\Admin\Desktop\final\adapters\Deepfake_adapter\vision_adapter_toplevel",
    custom_heads_path=r"C:\Users\Admin\Desktop\final\adapters\Deepfake_adapter\custom_heads\custom_heads.safetensors",
)
t0 = time.perf_counter()
sel = svc.select("anomaly")
print("SELECT:", sel, f"{time.perf_counter()-t0:.1f}s")
n = 0
t0 = time.perf_counter()
for chunk in svc.stream("XMEAS_7 reactor pressure high. Diagnose the fault.",
                        max_new_tokens=20, stub=False):
    if chunk.get("event") == "token":
        n += 1
        print(f"[{n}] {chunk['text']!r} +{chunk['latency_ms']}ms")
    elif chunk.get("event") == "generation_receipt":
        print("RECEIPT:", chunk)
print(f"TOTAL {time.perf_counter()-t0:.1f}s")
