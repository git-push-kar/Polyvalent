import sys, os, time, traceback
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

for task, prompt in [("general", "Say hello briefly."),
                     ("anomaly", "XMEAS_7 high. Diagnose."),
                     ("deepfake", "Describe this in one sentence.")]:
    print(f"===== select({task}) =====", flush=True)
    try:
        print(svc.select(task), flush=True)
    except Exception:
        traceback.print_exc()
        continue
    try:
        for chunk in svc.stream(prompt, max_new_tokens=6, stub=False):
            if chunk.get("event") == "token":
                print("TOK:", repr(chunk["text"]), flush=True)
            elif chunk.get("event") == "notice":
                print("NOTICE:", chunk["text"][:200], flush=True)
            elif chunk.get("event") == "generation_receipt":
                print("TOK/S:", chunk["tok_per_s"], flush=True)
    except Exception:
        traceback.print_exc()
print("REPRO DONE", flush=True)
