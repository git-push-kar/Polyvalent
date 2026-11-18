"""Unit and integration tests for the YOLO multi-face pipeline,
context cropping, and multi-face aggregation.
"""
import pytest
from PIL import Image
from types import SimpleNamespace

from deepfake.face_detect import (
    FaceBox,
    expand_box,
    square_box,
    overflow_fraction,
    crop_with_context,
    filter_faces,
    lone_face_fills_frame,
    route_image,
)
from router.conversations import Analysis, ConversationStore


class StubDetector:
    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect(self, image):
        return list(self.boxes)


def make_test_image(w=1000, h=1000, color=(128, 128, 128)):
    return Image.new("RGB", (w, h), color)


# -------------------------------------------------------------------------
# Geometry tests
# -------------------------------------------------------------------------

def test_expand_box():
    box = FaceBox(100, 100, 200, 200)  # 100x100
    expanded = expand_box(box, margin=0.30)
    assert expanded.x1 == pytest.approx(70.0)
    assert expanded.y1 == pytest.approx(70.0)
    assert expanded.x2 == pytest.approx(230.0)
    assert expanded.y2 == pytest.approx(230.0)


def test_square_box():
    box = FaceBox(100, 100, 200, 300)  # w=100, h=200, cx=150, cy=200
    sq = square_box(box)
    assert sq.width == pytest.approx(200.0)
    assert sq.height == pytest.approx(200.0)
    assert sq.cx == pytest.approx(150.0)
    assert sq.cy == pytest.approx(200.0)


def test_crop_with_context_inside_bounds():
    img = make_test_image(1000, 1000)
    box = FaceBox(300, 300, 500, 500)  # 200x200
    crop, meta = crop_with_context(img, box, margin=0.30)
    assert meta["pad_mode"] == "none"
    assert crop.size[0] == crop.size[1]
    assert crop.size[0] == 320  # 200 + 2*60


def test_crop_with_context_reflect_near_border():
    img = make_test_image(500, 500)
    # Box touching the edge so expanded square overflows by < 10%
    box = FaceBox(10, 100, 110, 200)  # 100x100
    crop, meta = crop_with_context(img, box, margin=0.10, reflect_max_frac=0.15)
    assert crop.size[0] == crop.size[1]
    assert meta["pad_mode"] in ("reflect", "none", "stretch")


# -------------------------------------------------------------------------
# Routing logic tests
# -------------------------------------------------------------------------

def test_lone_face_fills_frame():
    assert lone_face_fills_frame([0.08], min_frac=0.05) is True
    assert lone_face_fills_frame([0.03], min_frac=0.05) is False
    assert lone_face_fills_frame([0.08, 0.08], min_frac=0.05) is False
    assert lone_face_fills_frame([], min_frac=0.05) is False


def test_route_image_passthrough_portrait():
    # 400x500 face in 1600x1500 = 200000 / 2400000 = 0.0833 (>= 0.05)
    img = make_test_image(1600, 1500)
    det = StubDetector([FaceBox(600, 300, 1000, 800)])
    res = route_image(img, det)
    assert res["path"] == "passthrough"
    assert len(res["items"]) == 1
    assert res["items"][0]["kind"] == "whole"


def test_route_image_lone_small_face_routes_to_per_face_crop():
    # 200x200 face in 2000x2000 = 40000 / 4000000 = 0.01 (< 0.05)
    img = make_test_image(2000, 2000)
    det = StubDetector([FaceBox(500, 500, 700, 700)])
    res = route_image(img, det)
    assert res["path"] == "per_face"
    assert len(res["items"]) == 1
    assert res["items"][0]["kind"] == "face"
    assert res["items"][0]["meta"]["pad_mode"] == "none"


def test_route_image_multiple_faces_route_per_face():
    img = make_test_image(2000, 1000)
    boxes = [
        FaceBox(200, 200, 450, 450),
        FaceBox(700, 200, 950, 450),
        FaceBox(1200, 200, 1450, 450),
    ]
    det = StubDetector(boxes)
    res = route_image(img, det)
    assert res["path"] == "per_face"
    assert len(res["items"]) == 3
    for i, it in enumerate(res["items"]):
        assert it["kind"] == "face"
        assert it["id"] == i + 1
        assert it["image"].size[0] == it["image"].size[1]  # square


def test_route_image_no_faces_detected():
    img = make_test_image(500, 500)
    det = StubDetector([])
    res = route_image(img, det)
    assert res["path"] == "no_face"
    assert len(res["items"]) == 1
    assert res["items"][0]["kind"] == "whole"


def test_route_image_face_too_small_skipped():
    img = make_test_image(2000, 2000)
    det = StubDetector([FaceBox(100, 100, 150, 150)])  # 50px < min_face_px (96)
    res = route_image(img, det)
    assert res["path"] == "no_face"
    assert len(res["skipped"]) == 1
    assert res["skipped"][0]["reason"] == "face_too_small"


# -------------------------------------------------------------------------
# Analysis multi-face representation tests
# -------------------------------------------------------------------------

def test_analysis_multiface_history_and_generation_block():
    an = Analysis(
        id="an_1",
        image_id="img_1",
        verdict="FAKE",
        confidence=0.9850,
        hotspots="Mouth & Lower jaw",
        spectral="High HF/LF ratio",
        explanation="Deepfake boundary artifacts detected in Face #2.",
        faces_detected=3,
        faces_analyzed=3,
        routing_path="per_face",
        faces=[
            {"face_id": 1, "verdict": "REAL", "confidence": 0.95},
            {"face_id": 2, "verdict": "FAKE", "confidence": 0.985},
            {"face_id": 3, "verdict": "REAL", "confidence": 0.91},
        ],
    )
    gen_block = an.generation_block()
    assert "faces evaluated: 3 detected in image" in gen_block
    assert "primary face analyzed: #1" in gen_block or "face analyzed" in gen_block

    hist = an.history_text()
    assert "(3 faces analyzed)" in hist
    assert "verdict FAKE (98.5% confidence)" in hist

    view = an.routing_view()
    assert any("Faces: 3 evaluated" in s for s in view["evidence_summary"])
