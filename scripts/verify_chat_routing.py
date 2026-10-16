import sys, os, json, base64
sys.path.insert(0, r"C:\Users\Admin\Desktop\final")
os.environ.pop("FINAL_STUB_GENERATION", None)  # REAL path for this suite
from fastapi.testclient import TestClient
import api.server as srv

print("threshold:", srv.get_laya().min_confidence)
c = TestClient(srv.app)

with open(r"C:\Users\Admin\Desktop\LY_C68\deepfake_project\test_images\1.jpg",
          "rb") as fh:
    real_b64 = base64.b64encode(fh.read()).decode()

tests = [
    ("Please tell me if this image is fake or real", {}, "notice"),
    ("is this face real or fake", {"image_base64": real_b64}, "verdict"),
    ("is this face real or fake", {"image_base64": "AAAA"}, "notice"),
]
ok = True
for msg, extra, want in tests:
    kinds = []
    with c.stream("POST", "/chat/stream",
                  json={"message": msg, "max_new_tokens": 30, **extra}) as s:
        for line in s.iter_lines():
            if line and line.startswith("data:"):
                kinds.append(json.loads(line[6:])["event"])
                if len(kinds) > 4 and want == "verdict":
                    break
    good = kinds and kinds[0] == want and kinds[-1] in ("done", "token")
    if want == "verdict":
        good = kinds and kinds[0] == "verdict"
    ok &= good
    print(("PASS" if good else "FAIL"), repr(msg[:45]), "->", kinds[:6])
print("ALL OK" if ok else "FAILURES PRESENT")
