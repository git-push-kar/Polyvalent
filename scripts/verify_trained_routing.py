"""Routing-level verification for the fine-tuned BASE/DEEPFAKE router.

Runs the real checkpoint (models/laya_router) over the five-field state built
by ConversationStore, one case per requirement:

  1 image + ordinary visual question   -> BASE (general), image available
  2 image + authenticity request       -> DEEPFAKE
  3 follow-up about a recorded verdict -> BASE, saved analysis in context
  4 explicit authenticity rerun        -> DEEPFAKE
  5 deep_thinking bypass               -> reasoning (server-owned; asserted here)

No generation: this only exercises routing and the routing/generation split.
Run with the venv that has `laya` installed:

  C:/Users/Admin/Desktop/anomaly/venv/Scripts/python.exe scripts/verify_trained_routing.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from router.conversations import Analysis, ConversationStore  # noqa: E402
from router.laya_gate import serialize  # noqa: E402
from router.trained_router import TrainedLayaRouter  # noqa: E402


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (96, 96), (120, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    store = ConversationStore()
    router = TrainedLayaRouter()
    conv = "chat_verify"
    raw = _png()
    failures = []

    def ask(label: str, message: str, *, expect: str, image: bool = False) -> dict:
        if image:
            store.add_image(conv, None, raw, media_type="image/png")
        state = store.routing_state(conv, message)
        decision = router.route(state)
        gate = (decision.extra or {}).get("gate") or {}
        print(f"\n--- {label}")
        print("router input:", serialize(state)[:600])
        print("gate output :", json.dumps(
            {k: gate.get(k) for k in ("route", "probability", "probabilities")}))
        print("decision    :", decision.task, "|", decision.reason,
              "| tokens", (decision.extra or {}).get("input_tokens"),
              "/", (decision.extra or {}).get("input_budget"))
        if decision.task != expect:
            failures.append(f"{label}: expected {expect}, got {decision.task}")
        return decision

    ask("case 1 image + ordinary visual question",
        "Describe the person in this image", expect="general", image=True)
    ask("case 2 image + authenticity request",
        "Is the face in this photo real or fake?", expect="deepfake")

    asset = store.get(conv).active_asset()
    analysis = store.record_analysis(conv, Analysis(
        id="an_1", image_id=asset.id, verdict="FAKE", confidence=0.91,
        hotspots="Eyes & Facial midline", spectral="high-frequency residue",
        explanation="Blending seams around the eyes."))
    store.add_turn(conv, "user", "Is the face in this photo real or fake?",
                   kind="analysis")
    store.add_turn(conv, "assistant", analysis.history_text(),
                   kind="analysis")

    follow = ask("case 3 follow-up about the verdict",
                 "Explain that verdict", expect="general")
    last = (follow.extra or {}).get("last_analysis") or {}
    print("last_analysis keys:", sorted(last))
    if sorted(last) != ["asset_id", "confidence", "evidence_summary", "verdict"]:
        failures.append("last_analysis does not match the archive structure")
    ask("case 4 explicit authenticity rerun",
        "Check the same photo again and rerun the real/fake analysis",
        expect="deepfake")

    # Regression: a neutral greeting after a verdict must not re-fire DEEPFAKE.
    # The saved analysis replaces the raw forensics exchange in routing history.
    for greeting in ("hello", "kaisa h"):
        ask(f"greeting after verdict: {greeting!r}", greeting, expect="general")
    ask("explicit rerun still deepfake",
        "check it again", expect="deepfake")

    print("\n--- generation context for case 3")
    ctx = store.generation_context(conv, image_id=asset.id)
    print("history turns:", len(ctx["history"]))
    print("analysis block:", ctx["analysis_block"][:400])

    print("\n--- truncation check (history must drop, message must survive)")
    long_conv = "chat_long"
    long_image = store.add_image(long_conv, None, raw)
    for i in range(8):
        store.add_turn(long_conv, "user", f"earlier turn {i} " + "context " * 200)
    message = "Explain that verdict with the recorded analysis"
    state = store.routing_state(long_conv, message)
    decision = router.route(state)
    extra = decision.extra or {}
    print("tokens:", extra.get("input_tokens"), "/", extra.get("input_budget"),
          "history turns kept:", extra.get("history_turns"),
          "dropped:", extra.get("history_dropped"),
          "message intact:", message in (extra.get("router_input") or ""))
    if message not in (extra.get("router_input") or ""):
        failures.append("truncation dropped the message")

    print("\n--- guard: DEEPFAKE wanted, image not usable -> BASE, no pipeline")
    ghost = "chat_ghost"
    store.set_active_image(ghost, "img_missing")
    decision = router.route(store.routing_state(
        ghost, "Is this photo real or fake?"))
    print("decision:", decision.task, "|", decision.reason)
    # The checkpoint itself reads the asset list: with no supported image it
    # answers BASE. The router's own guard is the second layer for the case
    # where a head says DEEPFAKE anyway. Either way: never the pipeline.
    if decision.task == "deepfake":
        failures.append("missing image still selected deepfake")

    print("\n--- case 5 deep_thinking is not part of the five-field input")
    state = store.routing_state(conv, "Explain that verdict")
    print("state keys:", sorted(state))
    if sorted(state) != ["active_image_id", "assets", "history",
                         "last_analysis", "message"]:
        failures.append("routing state keys drifted")

    print("\nRESULT:", "PASS" if not failures else "FAIL")
    for f in failures:
        print(" -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
