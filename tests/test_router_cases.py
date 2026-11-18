"""Comprehensive test suite for the fine-tuned LAYA router, two-verdict mechanism,
and multi-turn conversation context.

Verifies the 5 core cases + the two-verdict recent image reference case:
  1. Image + ordinary visual question -> LAYA consulted, BASE receives pixels.
  2. Image + authenticity request -> LAYA consulted, deepfake pipeline runs.
  3. Follow-up asking about that verdict -> BASE receives stored result and history.
  4. Explicit authenticity rerun -> deepfake pipeline reruns.
  5. Thinking ON -> LR directly; turning it OFF restores normal routing.
  6. Recent image in conversation + later verification request -> Two-verdict LAYA triggers DEEPFAKE.
"""
import base64
import io
import json
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

os.environ["FINAL_STUB_GENERATION"] = "1"

import api.server as srv
from router.conversations import ConversationStore, Analysis, is_verdict_question
from router.trained_router import TrainedLayaRouter
from router.laya_gate import LayaGate, build_state, serialize

client = TestClient(srv.app)


def _sample_png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 100, 50)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def fresh_state():
    srv._SERVICE = None
    srv._CONVERSATIONS = ConversationStore()
    srv._TRAINED = None
    yield
    srv._SERVICE = None
    srv._TRAINED = None


def stream_events(path, body):
    out = []
    with client.stream("POST", path, json=body) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def chat_stream(message, **extra):
    return stream_events("/chat/stream", {"message": message, **extra})


# =========================================================================
# Unit tests on LayaGate and TrainedLayaRouter with actual checkpoint
# =========================================================================

def test_laya_gate_availability():
    gate = LayaGate()
    assert gate.available(), f"Laya gate should be available: {gate.load_error}"


def test_case_1_image_and_ordinary_visual_question_routes_to_base():
    """Case 1: Image attached + ordinary question -> LAYA selects BASE."""
    router = TrainedLayaRouter()
    res = router.route({
        "message": "Describe the person in this image",
        "attached_now": True,
        "attached_asset": {"id": "img_1", "kind": "image", "availability": "available", "support": "supported"},
        "active_image_id": "img_1",
    })
    assert res.task == "general", f"Expected general (BASE), got {res.task}"
    assert res.router == "laya-finetuned"
    assert res.reason == "gate:finetuned-base"
    assert res.extra["gate"]["route"] == "BASE"


def test_case_2_image_and_authenticity_request_routes_to_deepfake():
    """Case 2: Image attached + authenticity check -> LAYA selects DEEPFAKE."""
    router = TrainedLayaRouter()
    res = router.route({
        "message": "Is this face real or fake?",
        "attached_now": True,
        "attached_asset": {"id": "img_1", "kind": "image", "availability": "available", "support": "supported"},
        "active_image_id": "img_1",
    })
    assert res.task == "deepfake", f"Expected deepfake, got {res.task}"
    assert res.reason == "gate:finetuned-deepfake"
    assert res.extra["gate"]["route"] == "DEEPFAKE"
    assert res.confidence >= 0.90


def test_case_3_followup_asking_about_verdict_routes_to_base():
    """Case 3: Follow-up asking about verdict -> LAYA selects BASE (no rerun)."""
    router = TrainedLayaRouter()
    res = router.route({
        "message": "Explain that verdict",
        "attached_now": False,
        "recent_asset": {"id": "img_1", "kind": "image", "availability": "available", "support": "supported"},
        "active_image_id": "img_1",
        "last_analysis": {
            "asset_id": "img_1",
            "verdict": "FAKE",
            "confidence": 0.985,
            "evidence_summary": ["Hotspots: eye, mouth", "Spectral: high frequency anomaly"]
        }
    })
    assert res.task == "general", f"Expected general (BASE), got {res.task}"
    assert res.reason == "gate:finetuned-base"


def test_case_4_explicit_rerun_routes_to_deepfake():
    """Case 4: Explicit rerun request -> LAYA selects DEEPFAKE."""
    router = TrainedLayaRouter()
    res = router.route({
        "message": "Check if this face is fake again",
        "attached_now": False,
        "recent_asset": {"id": "img_1", "kind": "image", "availability": "available", "support": "supported"},
        "active_image_id": "img_1",
        "last_analysis": {
            "asset_id": "img_1",
            "verdict": "FAKE",
            "confidence": 0.985,
            "evidence_summary": ["Hotspots: eye, mouth"]
        }
    })
    assert res.task == "deepfake", f"Expected deepfake for explicit rerun, got {res.task}"
    assert res.reason == "gate:two-verdict-deepfake-recent-image"


def test_case_6_recent_image_two_verdict_mechanism():
    """Two-verdict mechanism: recent image in chat, subsequent message asks for verification without re-attaching."""
    router = TrainedLayaRouter()
    # Message: "Is this real or fake?" without image attached now, but recent_asset is present
    res = router.route({
        "message": "Is this real or fake?",
        "attached_now": False,
        "recent_asset": {"id": "img_1", "kind": "image", "availability": "available", "support": "supported"},
        "active_image_id": "img_1",
    })
    assert res.task == "deepfake"
    assert res.reason == "gate:two-verdict-deepfake-recent-image"
    two = res.extra.get("two_verdict")
    assert two is not None
    assert two["without_image"]["route"] == "BASE"
    assert two["with_image"]["route"] == "DEEPFAKE"


# =========================================================================
# End-to-end conversation flows via /chat/stream
# =========================================================================

def test_e2e_full_lifecycle_cases():
    conv_id = "test_chat_full_flow"
    img_b64 = _sample_png_b64()

    # Turn 1: Image + ordinary visual question ("Describe the person in this image")
    evs_1 = chat_stream(
        "Describe the person in this image",
        conversation_id=conv_id,
        image_base64=img_b64,
        image_ids=["img_1"],
        active_image_id="img_1"
    )
    route_1 = next(e for e in evs_1 if e["event"] == "route")
    assert route_1["task"] == "general"
    ctx_1 = next(e for e in evs_1 if e["event"] == "context")
    assert ctx_1["image_id"] == "img_1"  # BASE receives pixels!

    # Turn 2: Follow-up question asking for verification referencing that image
    # (NO image re-attached)
    evs_2 = chat_stream(
        "Is this face real or fake?",
        conversation_id=conv_id
    )
    route_2 = next(e for e in evs_2 if e["event"] == "route")
    assert route_2["task"] == "deepfake"
    # Deepfake pipeline executed and emitted verdict
    verdict_ev = next((e for e in evs_2 if e["event"] == "verdict"), None)
    assert verdict_ev is not None
    saved_ev = next((e for e in evs_2 if e["event"] == "analysis_saved"), None)
    assert saved_ev is not None

    # Check conversation store history: faithful assistant representation
    store = srv.get_conversations()
    conv = store.get(conv_id)
    assert conv is not None
    assert len(conv.turns) == 4  # 2 user + 2 assistant
    assert "verdict" in conv.turns[-1]["content"].lower()

    # Turn 3: Follow-up asking about that verdict ("Explain that verdict")
    evs_3 = chat_stream(
        "Explain that verdict",
        conversation_id=conv_id,
        deep_thinking=False
    )
    route_3 = next(e for e in evs_3 if e["event"] == "route")
    assert route_3["task"] == "general"  # BASE selected!
    ctx_3 = next(e for e in evs_3 if e["event"] == "context")
    assert ctx_3["analysis_supplied"] is True  # Stored result supplied!
    assert ctx_3["image_id"] is None  # Pixels NOT passed to avoid re-inventing forensic evidence

    # Turn 4: Explicit authenticity rerun
    evs_4 = chat_stream(
        "Run the deepfake check again on this face",
        conversation_id=conv_id
    )
    route_4 = next(e for e in evs_4 if e["event"] == "route")
    assert route_4["task"] == "deepfake"

    # Turn 5: Thinking ON -> LR directly
    evs_5 = chat_stream(
        "Solve this logic problem step by step",
        conversation_id=conv_id,
        deep_thinking=True
    )
    route_5 = next(e for e in evs_5 if e["event"] == "route")
    assert route_5["task"] == "reasoning"
    assert "manual:deep-thinking" in route_5["reason"]

    # Turn 6: Thinking OFF -> normal routing restored
    evs_6 = chat_stream(
        "Hello, how are you today?",
        conversation_id=conv_id,
        deep_thinking=False
    )
    route_6 = next(e for e in evs_6 if e["event"] == "route")
    assert route_6["task"] == "general"
    assert "gate:finetuned-base" in route_6["reason"]


def test_screenshot_2_flow():
    """Screenshot 2:
    1. 'do i look real' with image -> DEEPFAKE
    2. 'mind explaining why' -> BASE with analysis_supplied=True, attach_pixels=False
    3. 'does this image look fake' -> DEEPFAKE
    """
    conv_id = "test_screenshot_2"
    img_b64 = _sample_png_b64()

    # Turn 1: 'do i look real' with image -> DEEPFAKE
    evs_1 = chat_stream(
        "do i look real",
        conversation_id=conv_id,
        image_base64=img_b64,
        image_ids=["img_1"],
        active_image_id="img_1"
    )
    route_1 = next(e for e in evs_1 if e["event"] == "route")
    assert route_1["task"] == "deepfake"
    assert next((e for e in evs_1 if e["event"] == "verdict"), None) is not None

    # Turn 2: 'mind explaining why' -> BASE, analysis supplied, no pixels passed
    evs_2 = chat_stream(
        "mind explaining why",
        conversation_id=conv_id,
        deep_thinking=False
    )
    route_2 = next(e for e in evs_2 if e["event"] == "route")
    assert route_2["task"] == "general"
    ctx_2 = next(e for e in evs_2 if e["event"] == "context")
    assert ctx_2["analysis_supplied"] is True
    assert ctx_2["image_id"] is None  # no pixel re-feed

    # Turn 3: 'does this image look fake' -> DEEPFAKE
    evs_3 = chat_stream(
        "does this image look fake",
        conversation_id=conv_id
    )
    route_3 = next(e for e in evs_3 if e["event"] == "route")
    assert route_3["task"] == "deepfake"


def test_screenshot_3_flow_anaphora_resolution():
    """Screenshot 3:
    1. 'what do u think about this' with image -> BASE with pixels
    2. 'does it look fake' (no image attached) -> DEEPFAKE via two-verdict + anaphora resolution
    """
    conv_id = "test_screenshot_3"
    img_b64 = _sample_png_b64()

    # Turn 1: 'what do u think about this' with image -> BASE
    evs_1 = chat_stream(
        "what do u think about this",
        conversation_id=conv_id,
        image_base64=img_b64,
        image_ids=["img_1"],
        active_image_id="img_1"
    )
    route_1 = next(e for e in evs_1 if e["event"] == "route")
    assert route_1["task"] == "general"
    ctx_1 = next(e for e in evs_1 if e["event"] == "context")
    assert ctx_1["image_id"] == "img_1"  # BASE receives pixels!

    # Turn 2: 'does it look fake' (no image attached now, references prior image)
    evs_2 = chat_stream(
        "does it look fake",
        conversation_id=conv_id
    )
    route_2 = next(e for e in evs_2 if e["event"] == "route")
    assert route_2["task"] == "deepfake", f"Expected DEEPFAKE for 'does it look fake', got {route_2['task']}"
    assert route_2["reason"] == "gate:two-verdict-deepfake-recent-image"
    assert next((e for e in evs_2 if e["event"] == "verdict"), None) is not None



def test_return_to_portrait_after_topic_change():
    conv_id = "return_to_portrait"
    def turn(message, **kwargs):
        events = chat_stream(message, conversation_id=conv_id, **kwargs)
        assert events[-1]["event"] == "done"
        return {event["event"]: event for event in events}

    first = turn("Describe this portrait", image_base64=_sample_png_b64())
    image_id = first["context"]["image_id"]
    assert image_id and first["adapter"]["task"] == "general"
    for question in ("What is the capital of France?", "What is 3 + 5?",
                     "Name two oceans.", "What is a prime number?"):
        assert turn(question)["adapter"]["task"] == "general"

    visual = turn("What color was the shirt in the earlier portrait?")
    assert visual["adapter"]["task"] == "general"
    assert visual["context"]["image_id"] == image_id
    check = turn("Going back to the portrait, was it AI-generated?")
    assert check["adapter"]["task"] == "deepfake"
    assert check["analysis_saved"]["image_id"] == image_id

    missing = turn("Is this image fake?", active_image_id="missing")
    assert "verdict" not in missing
    assert missing.get("adapter", {}).get("task") != "deepfake"
