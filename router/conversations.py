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
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_IMAGE_BYTES = 12 * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
ROUTING_TURN_CHARS = 700
ROUTING_HISTORY_TURNS = 6
GENERATION_HISTORY_TURNS = 6
RECENT_IMAGE_TURNS = 6

_VERDICT_FOLLOWUP_WORDS = (
    "verdict", "analysis", "forensic", "result", "score", "confidence",
    "why fake", "why real", "why is it fake", "why is it real",
    "why did you say", "explain that", "explain why", "explain the",
    "evidence", "hotspot", "spectral", "findings", "how do you know",
    "reason for", "what makes you say", "how did you classify", "classify",
    "classification",
)

_EXPLANATION_FOLLOWUP_RE = re.compile(
    r'\b(explain|explaining|explanation|elaborate|elaborating|clarify|clarification|details|detail|breakdown|why|how so|how come|reason)\b',
    re.I
)


def is_verdict_question(message: str) -> bool:
    """True if message asks to explain or discuss a recorded forensic verdict."""
    low = (message or "").lower().strip()
    if any(w in low for w in _VERDICT_FOLLOWUP_WORDS):
        return True
    if _EXPLANATION_FOLLOWUP_RE.search(low):
        return True
    return False


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
    added_turn: int = 0  # len(conversation.turns) when the image arrived

    def routing_view(self) -> Dict[str, Any]:
        """Exactly the asset shape the Router repository trained on."""
        return {
            "id": self.id,
            "kind": self.kind,
            "availability": self.availability,
            "support": self.support,
        }

    @property
    def usable(self) -> bool:
        return self.availability == "available" and self.support == "supported" and len(self.data) > 0


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
    faces_detected: int = 1
    faces_analyzed: int = 1
    routing_path: str = "passthrough"
    faces: List[Dict[str, Any]] = field(default_factory=list)

    def routing_view(self) -> Dict[str, Any]:
        """`last_analysis` in the Router archive's exact field structure.

        The reviewed training rows use four keys only:
        `{asset_id, confidence, evidence_summary, verdict}`.
        """
        summary: List[str] = []
        if self.faces_analyzed > 1:
            summary.append(f"Faces: {self.faces_analyzed} evaluated")
        if self.hotspots:
            summary.append(f"Hotspots: {self.hotspots[:160]}")
        if self.spectral:
            summary.append(f"Spectral: {self.spectral[:160]}")
        return {
            "asset_id": self.image_id,
            "verdict": self.verdict,
            "confidence": round(float(self.confidence), 4),
            "evidence_summary": summary,
        }

    def generation_block(self) -> str:
        """What BASE is told about a verdict it must explain, not re-derive."""
        lines = [
            "[Recorded deepfake analysis — already computed, do not re-run]",
            f"image: {self.image_id}",
            f"verdict: {self.verdict}",
            f"classifier confidence: {self.confidence:.4f}",
        ]
        if self.faces_analyzed > 1:
            lines.append(
                f"faces evaluated: {self.faces_analyzed} detected in image (primary face analyzed: #{self.faces[0].get('face_id', 1) if self.faces else 1})"
            )
        lines.extend([
            f"hotspot regions: {self.hotspots or 'not reported'}",
            f"spectral note: {self.spectral or 'not reported'}",
        ])
        if self.explanation.strip():
            lines.append(f"explanation: {self.explanation.strip()[:900]}")
        lines.append(
            "Use ONLY these recorded findings; do not claim new measurements or new forensic evidence."
        )
        return "\n".join(lines)

    def history_text(self) -> str:
        """Faithful assistant-history representation of the result."""
        faces_note = f" ({self.faces_analyzed} faces analyzed)" if self.faces_analyzed > 1 else ""
        text = (
            f"Deepfake analysis of {self.image_id}{faces_note}: verdict {self.verdict} "
            f"({self.confidence * 100:.1f}% confidence). "
            f"Hotspots: {self.hotspots or 'n/a'}. "
            f"Spectrum: {self.spectral or 'n/a'}."
        )
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
        return asset if (asset and asset.usable) else None

    def recent_asset(self, max_turns: int = RECENT_IMAGE_TURNS) -> Optional[Asset]:
        """Returns the most recent usable image if added within the recent turn window."""
        # Check active asset first
        active = self.active_asset()
        if active and active.usable:
            if len(self.turns) - active.added_turn <= max_turns:
                return active

        # Otherwise check the latest added usable asset
        for asset in reversed(list(self.assets.values())):
            if asset.usable and (len(self.turns) - asset.added_turn <= max_turns):
                return asset
        return None

    def recent_image_id(self, max_turns: int = RECENT_IMAGE_TURNS) -> Optional[str]:
        """The id of the active or recent image if within the recent window."""
        asset = self.recent_asset(max_turns)
        return asset.id if asset else None

    # -- turns ----------------------------------------------------------
    def add_turn(self, role: str, content: str, kind: str = "chat") -> None:
        text = (content or "").strip()
        if not text:
            return
        self.turns.append({
            "role": role if role in ("user", "assistant") else "user",
            "content": text,
            "ts": round(time.time(), 3),
            "kind": kind,
        })
        self.updated_at = time.time()

    def seed_history(self, history: List[Any]) -> None:
        """Client-supplied history for conversations the server has not seen."""
        for turn in history or []:
            if isinstance(turn, dict) and turn.get("content"):
                self.add_turn(turn.get("role", "user"), str(turn["content"])[:2000])

    def history(self, limit: int) -> List[Dict[str, str]]:
        """Generation view: the real transcript, analysis turns included."""
        return [
            {"role": t["role"], "content": t["content"]}
            for t in self.turns[-limit:]
        ]

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
        """Validate and store an upload; never raise on a bad image."""
        from PIL import Image

        with self._lock:
            conv = self.get_or_create(conversation_id)
            asset = Asset(
                id=asset_id or conv.next_asset_id(),
                data=raw,
                media_type=media_type or "",
                added_turn=len(conv.turns),
            )
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
                conv.assets[image_id] = Asset(
                    id=image_id, availability="missing", support="unsupported",
                    detail="image id referenced but not uploaded"
                )
                conv.active_image_id = image_id
            conv.updated_at = time.time()

    # -- views ----------------------------------------------------------
    def recent_image_id(self, conversation_id: str,
                        max_turns: int = RECENT_IMAGE_TURNS) -> Optional[str]:
        with self._lock:
            return self.get_or_create(conversation_id).recent_image_id(max_turns)

    def recent_asset(self, conversation_id: str,
                     max_turns: int = RECENT_IMAGE_TURNS) -> Optional[Asset]:
        with self._lock:
            return self.get_or_create(conversation_id).recent_asset(max_turns)

    def routing_payload(self, conversation_id: str, message: str, *,
                        attached_asset: Optional[Any] = None,
                        attached_now: bool = False) -> Dict[str, Any]:
        """Construct the payload for TrainedLayaRouter.
        
        Includes the current message, attachment status, and any recent image asset.
        """
        with self._lock:
            conv = self.get_or_create(conversation_id)
            recent = conv.recent_asset(RECENT_IMAGE_TURNS)
            target_asset = attached_asset if (attached_now and attached_asset) else recent
            active_id = target_asset.id if target_asset else conv.active_image_id
            analysis = conv.analysis_for_image(active_id) if active_id else conv.last_analysis()

            attached_view = (
                attached_asset.routing_view() if hasattr(attached_asset, "routing_view")
                else (attached_asset if isinstance(attached_asset, dict) else None)
            )
            recent_view = (
                recent.routing_view() if hasattr(recent, "routing_view")
                else (recent if isinstance(recent, dict) else None)
            )
            assets = [attached_view] if (attached_now and attached_view) else ([recent_view] if recent_view else [])

            return {
                "message": message or "",
                "attached_now": bool(attached_now),
                "attached_asset": attached_view,
                "recent_asset": recent_view,
                "assets": assets,
                "active_image_id": active_id,
                "last_analysis": analysis.routing_view() if analysis else None,
                "history": conv.history(ROUTING_HISTORY_TURNS),
            }

    def generation_context(self, conversation_id: str,
                           image_id: Optional[str] = None,
                           message: str = "") -> Dict[str, Any]:
        """History + saved analysis for the generator (separate from router context)."""
        with self._lock:
            conv = self.get_or_create(conversation_id)
            target_id = image_id or conv.active_image_id
            analysis = conv.analysis_for_image(target_id) if target_id else conv.last_analysis()
            has_analysis = analysis is not None
            is_followup = has_analysis and is_verdict_question(message)
            return {
                "history": conv.history(GENERATION_HISTORY_TURNS),
                "analysis": analysis,
                "analysis_block": analysis.generation_block() if has_analysis else "",
                "is_verdict_followup": is_followup,
            }

    def record_analysis(self, conversation_id: str, analysis: Analysis) -> Analysis:
        with self._lock:
            conv = self.get_or_create(conversation_id)
            conv.record_analysis(analysis)
            return analysis

    def add_turn(self, conversation_id: str, role: str, content: str,
                 kind: str = "chat") -> None:
        with self._lock:
            self.get_or_create(conversation_id).add_turn(role, content, kind=kind)
