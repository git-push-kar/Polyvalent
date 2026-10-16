import sys, os, json
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ["FINAL_STUB_GENERATION"] = "1"
from fastapi.testclient import TestClient
import api.server as srv

print("threshold:", srv.get_laya().min_confidence)
c = TestClient(srv.app)

# End-user expectations: domain queries stream tokens; chit-chat is
# answered by the base model (general); only sub-threshold garbage abstains.
tests = [
    ("Please tell me if this image is fake or real", {}, "token"),
    ("hi", {}, "token"),  # general -> base model greeting
    ("XMEAS_7 reactor pressure high, diagnose the fault", {}, "token"),
    ("is this face real or fake", {"image_base64": "AAAA"}, "token"),
]
ok = True
for msg, extra, want in tests:
    kinds = []
    with c.stream("POST", "/chat/stream",
                  json={"message": msg, "max_new_tokens": 6, **extra}) as s:
        for line in s.iter_lines():
            if line and line.startswith("data:"):
                kinds.append(json.loads(line[6:])["event"])
    good = want in kinds and "done" in kinds
    ok &= good
    print(("PASS" if good else "FAIL"), repr(msg[:45]), "->", kinds)

# Abstain path still exists when forced above zero-shot band.
srv.get_laya().min_confidence = 0.99
kinds = []
with c.stream("POST", "/chat/stream",
              json={"message": "xyzzy qqq", "max_new_tokens": 6}) as s:
    for line in s.iter_lines():
        if line and line.startswith("data:"):
            kinds.append(json.loads(line[6:])["event"])
good = kinds == ["notice", "done"]
ok &= good
print(("PASS" if good else "FAIL"), "forced-abstain ->", kinds)
print("ALL OK" if ok else "FAILURES PRESENT")
