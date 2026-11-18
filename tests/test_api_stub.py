"""API in stub mode: routing -> lazy adapter swap events -> streamed reply.

No weights needed (FINAL_STUB_GENERATION=1). If the `laya` package is not
installed, the router abstains and the support guard decides from text
cues, which is exactly the fallback path production uses.
"""
import base64
import json
import os

os.environ["FINAL_STUB_GENERATION"] = "1"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.server as srv  # noqa: E402

client = TestClient(srv.app)


@pytest.fixture(autouse=True)
def fresh_service():
    srv._SERVICE = None
    srv._CONVERSATIONS = srv.ConversationStore()
    yield


def events(path, body):
    out = []
    with client.stream("POST", path, json=body) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def chat(message, **extra):
    return events("/chat/stream", {"message": message, **extra})


def adapter(evs):
    return next(e for e in evs if e["event"] == "adapter")


def _png_b64() -> str:
    """A real, decodable image: attachments are validated before routing."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 120, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _FakeTrainedRouter:
    """Deterministic stand-in: these tests exercise swap mechanics, not LAYA."""

    def __init__(self, mapping):
        self.mapping = mapping

    def route(self, state):
        task = self.mapping.get(state.get("message"), "general")
        return srv.RouterDecision(
            router="laya-finetuned", task=task, confidence=1.0,
            reason=f"test:{task}", latency_ms=0.0,
            extra={"gate": {"route": "DEEPFAKE" if task == "deepfake" else "BASE"},
                   "router_input": json.dumps(state, sort_keys=True),
                   "input_tokens": 1, "input_budget": 374,
                   "history_turns": len(state.get("history") or []),
                   "history_dropped": 0})


@pytest.fixture(autouse=True)
def fake_router():
    srv._TRAINED = _FakeTrainedRouter({})
    yield
    srv._TRAINED = None


def test_task_changes_swap_adapters():
    srv._TRAINED = _FakeTrainedRouter({
        "is this face real or fake?": "deepfake",
    })
    r = adapter(chat("Prove it step by step.", deep_thinking=True))
    assert r["task"] == "reasoning" and r["loaded"] == ["base", "reasoning"]

    d = adapter(chat("is this face real or fake?", image_base64=_png_b64()))
    assert d["task"] == "deepfake"
    assert d["loaded"] == ["deepfake"] and "reasoning" in d["unloaded"]

    g = adapter(chat("What is the capital of France?"))
    assert g["task"] == "general" and "deepfake" in g["unloaded"]
    assert g["loaded"] == ["base"]

    st = client.get("/adapters/status").json()
    assert st["task"] == "general" and st["lora"] is None


def test_text_only_deepfake_request_does_not_load_the_pipeline():
    """Even when the router asks for DEEPFAKE, no pixels means no pipeline."""
    srv._TRAINED = _FakeTrainedRouter({
        "Check whether this face photo is real or fake": "deepfake",
    })
    chat("Solve this logic puzzle step by step: if A then B; A; therefore?",
         deep_thinking=True)
    evs = chat("Check whether this face photo is real or fake")
    assert not any(e["event"] == "adapter" for e in evs)
    assert any(e["event"] == "notice" for e in evs)
    assert client.get("/adapters/status").json()["lora"] == "reasoning"


def test_anomaly_tab_end_to_end():
    sc = client.get("/anomaly/scenarios").json()["scenarios"]
    faulty = next(s for s in sc if s["fault"])
    tl = client.post("/anomaly/run", json={"scenario_id": faulty["id"]}).json()
    assert tl["n"] == len(tl["scores"]) == len(tl["z"])
    ev = tl["events"][0]
    evs = events("/anomaly/reason", {"evidence": ev["evidence"],
                                     "values": ev["values"]})
    assert next(e for e in evs if e["event"] == "route")["task"] == "anomaly"
    assert adapter(evs)["loaded"] == ["base", "anomaly"]
    text = "".join(e["text"] for e in evs if e["event"] == "token")
    assert json.loads(text)["affected_subsystem"]
    fu = events("/anomaly/reason", {"evidence": ev["evidence"],
                                    "values": ev["values"],
                                    "question": "Which sensor moved first?"})
    assert adapter(fu)["loaded"] == []  # anomaly LoRA reused
    assert fu[-1]["event"] == "done"


def test_closing_stream_closes_sync_generator_on_early_exit():
    # Regression: Starlette never closes a sync body iterator on client
    # disconnect, which left `service.lease()` holding the model lock.
    import anyio

    closed = []

    def gen():
        try:
            for i in range(100):
                yield str(i)
        finally:
            closed.append(True)

    async def consume_one_then_disconnect():
        it = srv._closing(gen())
        async for _ in it:
            break
        await it.aclose()

    anyio.run(consume_one_then_disconnect)
    assert closed == [True]
