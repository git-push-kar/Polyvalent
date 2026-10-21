"""End-to-end chat routing regression suite (loads the real models).

Covers the routing failures reported in the UI:
  * plain conversation must never be hijacked by a specialist adapter
  * text-only deepfake requests ask for an image (and do not load the wrapper)
  * an image request yields a verdict
  * the next general turn returns to the bare base (no adapter stickiness)

Run with the uvicorn server STOPPED (this loads a second model copy):
    & $py scripts\\verify_chat_routing.py
"""
import base64
import json
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)  # REAL path for this suite

from fastapi.testclient import TestClient

import api.server as srv

print("threshold:", srv.get_laya().min_confidence, flush=True)
client = TestClient(srv.app)

IMAGE = (r"C:\Users\Admin\Desktop\LY_C68\deepfake_project"
         r"\test_images\1.jpg")
with open(IMAGE, "rb") as fh:
    REAL_B64 = base64.b64encode(fh.read()).decode()


def first_events(message, extra=None, limit=3):
    kinds = []
    body = {"message": message, "max_new_tokens": 64, **(extra or {})}
    with client.stream("POST", "/chat/stream", json=body) as resp:
        for line in resp.iter_lines():
            if line and line.startswith("data:"):
                kinds.append(json.loads(line[6:])["event"])
                if len(kinds) >= limit:
                    break
    return kinds


CASES = [
    ("Hi", {}, "token"),
    ("What is your name", {}, "token"),
    ("what is your nam", {}, "token"),
    ("my name is pushkar", {}, "token"),
    ("Write a Python function that reverses a string.", {}, "token"),
    ("What is the capital of France?", {}, "token"),
    ("Please tell me if this image is fake or real", {}, "notice"),
    ("is this face real or fake", {"image_base64": REAL_B64}, "verdict"),
    ("is this face real or fake", {"image_base64": "AAAA"}, "notice"),
    ("Hello again", {}, "token"),  # after the deepfake/image turns: base again
]

ok = True
for message, extra, want in CASES:
    kinds = first_events(message, extra)
    good = bool(kinds) and kinds[0] == want
    ok &= good
    print(("PASS" if good else "FAIL"), repr(message[:50]), "->", kinds,
          flush=True)
print("ALL OK" if ok else "FAILURES PRESENT", flush=True)
