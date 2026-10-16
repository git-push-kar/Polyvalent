"""Smoke: dual routing receipts + per-step latency + token streaming shape."""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ["FINAL_STUB_GENERATION"] = "1"

from router import LayaRouter, StaticRouter

static = StaticRouter()
cases = {
    "image": {"image_bytes": "AAA", "text": "is this face real?"},
    "sensor": {"values": [0.1] * 52, "text": "XMEAS_7 pressure high, XMV_3 stuck"},
    "portfolio": {"holdings": {"AAPL": 10}, "text": "rebalance for risk"},
    "vague": {"text": "hello there"},
}
for name, payload in cases.items():
    d = static.route(payload)
    print(f"[static/{name}]", json.dumps(d.to_dict()))

# Laya path: may route or abstain depending on installed package +
# confidence — either is fine. What must NEVER happen is static logic
# leaking in (no modality=/keyword= reasons, no silent substitution).
laya = LayaRouter()
d = laya.route({"text": "XMEAS_7 pressure high"})
print("[laya/sensor]", json.dumps(d.to_dict()))
assert not d.reason.startswith(("modality=", "keyword=")), \
    "laya must never fall back to static logic"
if d.abstained:
    assert d.reason.startswith("laya-"), d.reason

# Streaming shape: routing_receipt -> selection -> tokens -> generation_receipt -> done
from api.server import _predict_stream
from api.server import PredictRequest

req = PredictRequest(payload=cases["sensor"], prompt="summarise", max_new_tokens=20)
events = []
for raw in _predict_stream("static", req):
    assert raw.startswith("data: "), raw[:20]
    events.append(json.loads(raw[len("data: "):]))
kinds = [e["event"] for e in events]
print("stream events:", kinds)
assert kinds[0] == "routing_receipt"
assert "selection_receipt" in kinds and kinds[-1] == "done"
toks = [e for e in events if e["event"] == "token"]
assert toks and all("latency_ms" in t and "elapsed_ms" in t for t in toks), "per-token latency required"
gen = next(e for e in events if e["event"] == "generation_receipt")
assert gen["tokens"] == len(toks) and "tok_per_s" in gen and "per_token_ms" in gen
print(f"tokens={len(toks)} tok_per_s={gen['tok_per_s']} prefill={gen['prefill_ms']} decode={gen['decode_ms']}")
print("SMOKE OK")
