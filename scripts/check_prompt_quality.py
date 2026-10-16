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

tests = [
    ("general", [], "hi", 30),
    ("anomaly", [], "XMEAS_7 reactor pressure high, XMV_3 valve stuck. Diagnose the fault.", 120),
]
for task, hist, msg, ntok in tests:
    print(f"===== {task}: {msg[:50]} =====", flush=True)
    print(svc.select(task), flush=True)
    prompt = svc.format_prompt(task, hist, msg)
    print("PROMPT HEAD:", repr(prompt[:200]), flush=True)
    t0 = time.perf_counter()
    out = []
    for chunk in svc.stream(prompt, max_new_tokens=ntok, stub=False,
                            repetition_penalty=1.1):
        if chunk.get("event") == "token":
            out.append(chunk.get("text", ""))
        elif chunk.get("event") == "notice":
            out.append(f"[NOTICE {chunk.get('text','')[:150]}]")
        elif chunk.get("event") == "generation_receipt":
            print("TOK/S:", chunk["tok_per_s"], "TOKENS:", chunk["tokens"], flush=True)
    print("ANSWER:", "".join(out)[:800], flush=True)
    print(f"({time.perf_counter()-t0:.1f}s total)", flush=True)
print("PROMPT-CHECK DONE", flush=True)
