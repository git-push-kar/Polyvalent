"""Server-side conversation state shared by routing and generation.

Two views of the same conversation, deliberately kept separate:

  routing_view()   the compact five-field state the fine-tuned LAYA router
                   was trained on (message, assets, active_image_id,
                   last_analysis, history) — metadata only, never pixels,
                   never a full transcript.
  generation_view() the recent turns plus the saved analysis text handed to
                   the selected generator (base / LR / deepfake).

Assets hold the actual uploaded bytes so a later turn can reuse the same image
for a new verdict without a re-upload. Analyses hold what the deepfake pipeline
actually produced (verdict, score, hotspot regions, spectral note, explanation)
linked to the conversation and the image, so a follow-up can explain recorded
findings instead of rerunning forensics.

All mutation goes through one lock: a conversation is read while requests are
in flight, and Streamlit/FastAPI may resume an SSE generator on another thread.
"""
from __future__ import annotations

import hashlib
import io
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_IMAGE_BYTES = 12 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
# A routing turn is a pointer to what was said, not a transcript: the router
# has a 512-token budget and the newest message must survive intact.
ROUTING_TURN_CHARS = 700
ROUTING_HISTORY_TURNS = 6
GENERATION_HISTORY_TURNS = 6


@dataclass
class Asset:
    """One uploaded image (routing metadata + the bytes generation needs)."""

    id: str
    data: bytes = b""
    kind: str = "image"
    availability: str = "available"
    support: str = "supported"
    media_type: str = ""
    sha256: str = ""
    detail: str = ""
    added_at: float = field(default_factory=time.time)

    def routing_view(self) -> Dict[str, Any]:
        """Exactly the asset shape the Router repository trained on."""
        return {"id": self.id, "kind": self.kind,
                "availability": self.availability, "support": self.support}

    @property
    def usable(self) -> bool:
        return self.availability == "available" and self.support == "supported"


@dataclass
class Analysis:
    """A completed deepfake run, stored for later explanation."""

    id: str
    image_id: str
    verdict: str
    confidence: float
    hotspots: str = ""
    spectral: str = ""
    explanation: str = ""
    model: str = "deepfake-wrapper"
    created_at: float = field(default_factory=time.time)

    def routing_view(self) -> Dict[str, Any]:
        """`last_analysis` in the Router archive's exact field structure.

        The reviewed training rows use four keys only:
        `{asset_id, confidence, evidence_summary, verdict}`. The long prose
        stays out of the router's input; the generation context carries it.
        """
        summary: List[str] = []
        if self.hotspots:
            summary.append(f"Hotspots: {self.hotspots[:160]}")
        if self.spectral:
            summary.append(f"Spectral: {self.spectral[:160]}")
        if self.explanation.strip():
            summary.append(f"Explanation: {self.explanation.strip()[:240]}")
        return {"asset_id": self.image_id, "verdict": self.verdict,
                "confidence": round(float(self.confidence), 4),
                "evidence_summary": summary}

    def generation_block(self) -> str:
        """What BASE is told about a verdict it must explain, not re-derive."""
        lines = [
            "[Recorded deepfake analysis — already computed, do not re-run]",
            f"image: {self.image_id}",
            f"verdict: {self.verdict}",
            f"classifier confidence: {self.confidence:.4f}",
            f"hotspot regions: {self.hotspots or 'not reported'}",
            f"spectral note: {self.spectral or 'not reported'}",
        ]
        if self.explanation.strip():
            lines.append(f"explanation: {self.explanation.strip()[:900]}")
        lines.append("Use ONLY these recorded findings; do not claim new "
                     "measurements or new forensic evidence.")
        return "\n".join(lines)

    def history_text(self) -> str:
        """Faithful assistant-history representation of the result."""
        text = (f"Deepfake analysis of {self.image_id}: verdict {self.verdict} "
                f"({self.confidence * 100:.1f}% confidence). "
                f"Hotspots: {self.hotspots or 'n/a'}. "
                f"Spectrum: {self.spectral or 'n/a'}.")
        if self.explanation.strip():
            text += f" {self.explanation.strip()[:600]}"
        return text


class Conversation:
    def __init__(self, conversation_id: str) -> None:
        self.id = conversation_id
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.turns: List[Dict[str, Any]] = []
        self.assets: Dict[str, Asset] = {}
        self.active_image_id: Optional[str] = None
        self.analyses: List[Analysis] = []
        self._asset_seq = 0

    # -- assets ---------------------------------------------------------
    def next_asset_id(self) -> str:
        while True:
            self._asset_seq += 1
            candidate = f"img_{self._asset_seq}"
            if candidate not in self.assets:
                return candidate

    def active_asset(self) -> Optional[Asset]:
        if not self.active_image_id:
            return None
        return self.assets.get(self.active_image_id)

    def usable_asset(self) -> Optional[Asset]:
        asset = self.active_asset()
        return asset if asset and asset.usable else None

    # -- turns ----------------------------------------------------------
    def add_turn(self, role: str, content: str, kind: str = "chat") -> None:
        text = (content or "").strip()
        if not text:
            return
        self.turns.append({"role": role if role in ("user", "assistant") else "user",
                           "content": text, "ts": round(time.time(), 3),
                           "kind": kind})
        self.updated_at = time.time()

    def seed_history(self, history: List[Any]) -> None:
        """Client-supplied history for conversations the server has not seen."""
        for turn in history or []:
            if isinstance(turn, dict) and turn.get("content"):
                self.add_turn(turn.get("role", "user"), str(turn["content"])[:2000])

    def history(self, limit: int) -> List[Dict[str, str]]:
        """Generation view: the real transcript, analysis turns included."""
        return [{"role": t["role"], "content": t["content"]}
                for t in self.turns[-limit:]]

    # -- analyses -------------------------------------------------------
    def record_analysis(self, analysis: Analysis) -> Analysis:
        self.analyses.append(analysis)
        self.updated_at = time.time()
        return analysis

    def last_analysis(self) -> Optional[Analysis]:
        return self.analyses[-1] if self.analyses else None

    def analysis_for_image(self, image_id: Optional[str]) -> Optional[Analysis]:
        for a in reversed(self.analyses):
            if image_id is None or a.image_id == image_id:
                return a
        return None


class ConversationStore:
    """Process-wide conversation state (survives adapter swaps by design)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conversations: Dict[str, Conversation] = {}

    # -- lifecycle ------------------------------------------------------
    def get_or_create(self, conversation_id: str) -> Conversation:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                conv = Conversation(conversation_id)
                self._conversations[conversation_id] = conv
            return conv

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._conversations.get(conversation_id)

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._conversations)

    # -- images ---------------------------------------------------------
    def add_image(self, conversation_id: str, asset_id: Optional[str],
                  raw: bytes, media_type: str = "") -> Asset:
        """Validate and store an upload; never raise on a bad image.

        A file that cannot be decoded (or is too large / the wrong kind) is
        kept as `availability=missing` or `support=unsupported`, so routing
        sees the real reason and the deepfake pipeline is never selected for
        it. The Router repository's `supported_asset_exists` contract is what
        the gate was trained with.
        """
        from PIL import Image

        with self._lock:
            conv = self.get_or_create(conversation_id)
            asset = Asset(id=asset_id or conv.next_asset_id(), data=raw,
                          media_type=media_type or "")
            if len(raw) > MAX_IMAGE_BYTES:
                asset.availability, asset.support = "available", "unsupported"
                asset.detail = f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
            elif not raw:
                asset.availability, asset.support = "missing", "unsupported"
                asset.detail = "empty upload"
            else:
                asset.sha256 = hashlib.sha256(raw).hexdigest()
                try:
                    img = Image.open(io.BytesIO(raw))
                    img.load()
                    asset.media_type = asset.media_type or Image.MIME.get(img.format, "")
                    if img.format not in SUPPORTED_IMAGE_FORMATS:
                        asset.support = "unsupported"
                        asset.detail = f"format {img.format} is not supported"
                except Exception as exc:  # corrupt / not an image
                    asset.availability, asset.support = "missing", "unsupported"
                    asset.detail = f"cannot decode image: {type(exc).__name__}"
            conv.assets[asset.id] = asset
            if conv.active_image_id is None or asset.usable:
                conv.active_image_id = asset.id
            conv.updated_at = time.time()
            return asset

    def set_active_image(self, conversation_id: str, image_id: str) -> None:
        with self._lock:
            conv = self.get_or_create(conversation_id)
            if image_id in conv.assets:
                conv.active_image_id = image_id
            else:
                # Referenced but never uploaded: record the gap instead of
                # silently pointing at another image.
                conv.assets[image_id] = Asset(
                    id=image_id, availability="missing", support="unsupported",
                    detail="image id referenced but not uploaded")
                conv.active_image_id = image_id
            conv.updated_at = time.time()

    # -- views ----------------------------------------------------------
    def routing_state(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """The exact five-field input, populated from server-side state.

        The deepfake exchange is *not* replayed raw into `history`. The
        checkpoint (trained with one short summary turn per prior exchange)
        reads a raw prior authenticity request as an operation still pending,
        so a neutral follow-up ("hello") re-fires DEEPFAKE. Once an analysis
        exists it is already carried by `last_analysis`, so routing history
        replaces that exchange with a neutral completion line. Generation
        history keeps the full transcript; only the routing view is compacted.
        """
        with self._lock:
            conv = self.get_or_create(conversation_id)
            assets = ([conv.assets[i].routing_view()
                       for i in conv.assets] if conv.assets else [])
            active = conv.active_image_id
            analysis = conv.analysis_for_image(active)
            last = analysis or conv.last_analysis()
            history = [
                {"role": t["role"],
                 "content": t["content"][:ROUTING_TURN_CHARS]}
                for t in conv.turns
                if t.get("kind") != "analysis"
            ][-ROUTING_HISTORY_TURNS:]
            if analysis is not None:
                history = (history + [{
                    "role": "assistant",
                    "content": ("I completed the authenticity check for "
                                f"{analysis.image_id}; the result is saved."),
                }])[-ROUTING_HISTORY_TURNS:]
            return {"message": message or "", "history": history,
                    "assets": assets, "active_image_id": active,
                    "last_analysis": last.routing_view() if last else None}

    def generation_context(self, conversation_id: str,
                           image_id: Optional[str] = None) -> Dict[str, Any]:
        """History + saved analysis for the generator (never for the router)."""
        with self._lock:
            conv = self.get_or_create(conversation_id)
            analysis = conv.analysis_for_image(image_id) if image_id else conv.last_analysis()
            return {"history": conv.history(GENERATION_HISTORY_TURNS),
                    "analysis": analysis,
                    "analysis_block": analysis.generation_block() if analysis else ""}

    def record_analysis(self, conversation_id: str, analysis: Analysis) -> Analysis:
        with self._lock:
            conv = self.get_or_create(conversation_id)
            conv.record_analysis(analysis)
            return analysis

    def add_turn(self, conversation_id: str, role: str, content: str,
                 kind: str = "chat") -> None:
        with self._lock:
            self.get_or_create(conversation_id).add_turn(role, content, kind=kind)
