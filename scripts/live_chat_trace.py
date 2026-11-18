"""Post one chat turn and print its SSE events compactly (for traces).

Usage:
  python scripts/live_chat_trace.py --message "..." [--image path.png]
        [--conversation chat_123] [--thinking] [--max-tokens 48]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/chat/stream"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--image")
    ap.add_argument("--conversation")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    body = {"message": args.message, "max_new_tokens": args.max_tokens}
    if args.conversation:
        body["conversation_id"] = args.conversation
    if args.thinking:
        body["deep_thinking"] = True
    if args.image:
        raw = Path(args.image).read_bytes()
        body["image_base64"] = base64.b64encode(raw).decode()
        body["image_ids"] = [Path(args.image).stem]

    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text_parts = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:].strip())
            kind = ev.get("event")
            if kind in ("routing", "route", "context", "adapter",
                        "analysis_saved", "verdict"):
                print(json.dumps(ev, ensure_ascii=False)[:1200])
            elif kind == "token":
                text_parts.append(ev.get("text", ""))
            elif kind == "thinking":
                pass
            elif kind == "notice":
                print("NOTICE:", ev.get("text"))
            elif kind == "done":
                print("DONE:", json.dumps(ev))
    answer = "".join(text_parts).strip()
    print("\nANSWER:", answer[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
