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
print(svc.select("deepfake"), flush=True)
img_path = r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\test_images\1.jpg"
raw = open(img_path, "rb").read()
print("bytes:", len(raw), flush=True)
n = 0
t0 = time.perf_counter()
for chunk in svc.analyze_image_stream(raw, max_new_tokens=120, stub=False):
    ev = chunk.get("event")
    if ev == "verdict":
        print("VERDICT:", {k: chunk[k] for k in ("verdict", "confidence", "hotspots", "telemetry_ms")}, flush=True)
        print("SPECTRAL:", str(chunk.get("spectral"))[:120], flush=True)
    elif ev == "token":
        n += 1
        if n <= 12:
            print("TOK:", repr(chunk.get("text", "")), flush=True)
    elif ev == "notice":
        print("NOTICE:", str(chunk.get("text"))[:200], flush=True)
    elif ev == "generation_receipt":
        print("TOK/S:", chunk["tok_per_s"], "TOKENS:", chunk["tokens"], flush=True)
print(f"TOTAL {time.perf_counter()-t0:.1f}s TOKENS={n}", flush=True)
print("DEEPFAKE-LIVE DONE", flush=True)
