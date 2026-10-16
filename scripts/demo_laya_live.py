"""Live Laya demo: raw SDK + our LayaRouter wrapper with per-step latency."""
import json
import sys
import time

sys.path.insert(0, r"C:\Users\Admin\Desktop\final")

print("== 1. Raw laya SDK ==")
# This box: stale metadata-less triton breaks torch._inductor import, which
# transformers' ModernBERT pulls in via @torch.compile at class-definition
# time. Inductor can't run on Windows-CPU anyway -> eager fallback.
import torch

try:
    import torch._inductor.runtime.hints  # noqa: F401  (exact submodule that breaks)
except Exception:
    _orig_compile = torch.compile

    def _eager_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda m: m
        return model

    torch.compile = _eager_compile
    print("(torch._inductor unusable here -> torch.compile set to eager no-op)")
t0 = time.perf_counter()
from laya import Router

router = Router()  # lazy: downloads selected checkpoint on first use
print(f"Router() constructed in {(time.perf_counter()-t0)*1000:.0f} ms (no checkpoint yet)")

questions = {
    "task": {
        "type": "choice",
        "instructions": "Route this request to the correct specialist adapter.",
        "criteria": {
            "deepfake": "face image, real/fake verdict, spectral or forgery artifacts",
            "anomaly": "TEP sensors XMEAS/XMV, pressure, valve, reactor, fault diagnosis",
            "portfolio": "holdings, returns, rebalance, risk, allocation",
            "general": "general question fitting no specialist",
        },
    }
}

cases = {
    "forensic": "Is this face image real or fake? Check for spectral forgery artifacts.",
    "tep": "XMEAS_7 reactor pressure high, XMV_3 valve stuck. Diagnose the fault.",
    "portfolio": "Rebalance my holdings to reduce risk and improve sharpe.",
    "vague": "hello there",
}

for name, text in cases.items():
    t0 = time.perf_counter()
    out = router.predict(text, questions)
    dt = (time.perf_counter() - t0) * 1000
    ans = out["answers"]["task"]
    print(f"[{name}] choice={ans['choice']} conf={ans.get('confidence')} "
          f"model={out['routing']['model']} total={dt:.0f}ms")

print()
print("== 2. Our LayaRouter wrapper (per-step latency, no static fallback) ==")
from router import LayaRouter

lr = LayaRouter()  # reuses cached checkpoint from part 1
payloads = {
    "forensic": {"text": cases["forensic"]},
    "tep": {"text": cases["tep"], "values": [0.1] * 52},
    "portfolio": {"text": cases["portfolio"]},
    "vague": {"text": cases["vague"]},
}
for name, payload in payloads.items():
    d = lr.route(payload)
    print(f"[{name}]", json.dumps(d.to_dict()))
print("DEMO OK")
