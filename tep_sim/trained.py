"""Bridge to the trained Layer-A detector of the C68-Anomaly pipeline.

`C68-Anomaly` ships the whole simulation pipeline (normal-only scaler, LSTM-AE,
event aggregation, evidence bridge), so the anomaly tab can replay a run through
the trained detector instead of the builtin z-score stand-in. Two defects in
that repository's artifacts are repaired here, at load time, without editing it:

  * Architecture metadata. `outputs/anomaly_detector/config.json` claims
    hidden 128 / latent 64 while `model.pt` holds hidden 32 / latent 16
    (`encoder.lstm.weight_ih_l0` = [128, 52] = 4*32). `from_artifacts` trusts the
    file, so it builds a model that can never accept the checkpoint. Shapes are
    inferred from the tensors instead.
  * Operating point. `threshold.json` (0.687) belongs to an older, larger
    model: the shipped 32/16 net scores the bundled fault-free runs at a mean of
    0.97 and a maximum of 1.53, i.e. it would alarm on every window. The
    threshold is re-derived with the repository's own method — a percentile of
    the score on fault-free data — over the test runs in `data/tep_test`.

Both values are reported in the timeline so a reader can tell a trained run
from the builtin detector and see which operating point was used. Point
`FINAL_ANOMALY_REPO` at a different checkout, `FINAL_ANOMALY_THRESHOLD` at a
threshold you want to pin, and `FINAL_TEP_TEST_DIR` at other fault-free runs.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dataset import Scenario, load

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_PERCENTILE = 99.0
_STATE: Dict[str, Any] = {"threshold": None, "source": None, "repo": None}
_LOCK = threading.Lock()


def repo_dir() -> Optional[Path]:
    """Locate the C68-Anomaly checkout (env override, then sibling folders)."""
    raw = (os.getenv("FINAL_ANOMALY_REPO") or "").strip().strip('"')
    candidates: List[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates += [_REPO.parent / "C68-Anomaly", _REPO.parent / "anomaly"]
    for cand in candidates:
        if not cand.is_absolute():
            cand = _REPO / cand
        if (cand / "main.py").is_file() and (cand / "anomaly_detection").is_dir():
            return cand.resolve()
    return None


# -- loader repair -----------------------------------------------------------

def _patch_loader() -> None:
    """Make `AnomalyDetector.from_artifacts` read the checkpoint, not the file."""
    from anomaly_detection import inference as inference_mod
    from anomaly_detection.lstm_autoencoder import build_autoencoder
    from anomaly_detection.threshold import load_threshold
    from preprocessing.scaler import load_scaler
    import torch

    detector_cls = inference_mod.AnomalyDetector
    if getattr(detector_cls, "_final_shape_loader", False):
        return

    @classmethod
    def from_artifacts(cls, model_dir, scaler_dir=None, threshold_dir=None,
                       device="cpu"):
        model_dir = Path(model_dir)
        scaler_dir = Path(scaler_dir or model_dir)
        threshold_dir = Path(threshold_dir or model_dir)
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        weights = state["model_state_dict"]

        hidden = int(weights["encoder.lstm.weight_ih_l0"].shape[0]) // 4
        latent = int(weights["encoder.project.weight"].shape[0])
        layers = 1 + max(int(k.rsplit("_l", 1)[1]) for k in weights
                         if k.startswith("encoder.lstm.weight_ih_l"))
        decoder_in = weights.get("decoder.lstm.weight_ih_l0")
        model = build_autoencoder(
            {"anomaly_detector": {"lstm": {
                "hidden_size": hidden, "num_layers": layers,
                "latent_dim": latent, "dropout": 0.05,
                "bidirectional_encoder": False}}},
            int(state["num_features"]), int(state["sequence_length"]),
            use_cell_proj="decoder.cell_proj.weight" in weights,
            decoder_input_dim=int(decoder_in.shape[1]) if decoder_in is not None else None,
        )
        model.load_state_dict(weights)
        return cls(model, load_scaler(scaler_dir),
                   load_threshold(threshold_dir), device=device)

    detector_cls.from_artifacts = from_artifacts
    detector_cls._final_shape_loader = True


# -- operating point ---------------------------------------------------------

def _fault_free_runs(limit: int = 3) -> List[np.ndarray]:
    """Fault-free arrays from the test set (the repo's own FAR calibration data)."""
    from .dataset import dataset_dir

    root = dataset_dir()
    if root is None:
        return []
    runs: List[np.ndarray] = []
    for path in sorted(root.glob("fault0_run*.csv"))[:limit]:
        try:
            runs.append(load(Scenario(id=path.stem, label=path.stem, fault=0,
                                      source="dataset", path=str(path))))
        except Exception as exc:  # unreadable file must not block the tab
            logger.debug("skipping %s for calibration: %s", path.name, exc)
    return runs


def _operating_point(app, repo: Path) -> Tuple[float, str]:
    pinned = (os.getenv("FINAL_ANOMALY_THRESHOLD") or "").strip()
    if pinned:
        return float(pinned), "FINAL_ANOMALY_THRESHOLD"
    scores: List[float] = []
    window, stride = int(app.window_size), int(app.stride)
    for run in _fault_free_runs():
        for t in range(0, len(run) - window + 1, stride):
            score, _ = app.detector.score_window(run[t:t + window])
            scores.append(float(score))
    if not scores:
        stored = float(app.detector.threshold.threshold)
        return stored, "threshold.json (no fault-free runs to recalibrate on)"
    derived = float(np.percentile(np.asarray(scores), _PERCENTILE))
    return derived, (f"p{_PERCENTILE:g} of {len(scores)} fault-free windows "
                     f"in data/tep_test")


def _build(repo: Path) -> Tuple[Any, float, str]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _patch_loader()
    from main import TEPApp  # type: ignore
    from utils import load_config  # type: ignore

    config_path = Path(os.getenv("FINAL_ANOMALY_CONFIG")
                       or (repo / "configs" / "config.yaml"))
    config = load_config(str(config_path))
    # Keep the training repo read-only: the simulator's event store goes here.
    store = _REPO / "logs" / "tep_sim"
    store.mkdir(parents=True, exist_ok=True)
    config.setdefault("events", {})["db_path"] = str(store / "events.db")
    with contextlib.chdir(repo):
        app = TEPApp(config, enable_llm=False)
        threshold, source = _operating_point(app, repo)
    app.detector.threshold.threshold = float(threshold)
    return app, threshold, source


# -- replay ------------------------------------------------------------------

def replay(scenario: Scenario, samples: Optional[np.ndarray] = None
           ) -> Dict[str, Any]:
    """Replay one run through the trained pipeline (fresh state per run)."""
    repo = repo_dir()
    if repo is None:
        raise FileNotFoundError("C68-Anomaly pipeline not found")

    with _LOCK:
        if _STATE["threshold"] is None:
            _STATE["threshold"], _STATE["source"] = None, None
        app, threshold, source = _build(repo)
        _STATE.update({"threshold": threshold, "source": source, "repo": str(repo)})

    x = load(scenario) if samples is None else np.asarray(samples)
    window = int(app.window_size)
    scores: List[float] = []
    flags: List[int] = []
    events: List[Dict[str, Any]] = []
    last = 0.0
    with contextlib.chdir(repo):
        for t, row in enumerate(x):
            result = app.process_sensor_stream(
                {"values": [float(v) for v in row], "sample_index": t,
                 "is_fault": bool(scenario.fault)}) or {}
            if result.get("window_score") is not None:
                last = float(result["window_score"])
            scores.append(round(last, 4))
            flags.append(int(bool(result.get("is_anomalous"))))
            if result.get("anomaly_detected") and result.get("event") is not None:
                captured = result["event"]
                for conv in ("model_dump", "dict", "to_dict"):
                    if not isinstance(captured, dict) and hasattr(captured, conv):
                        captured = getattr(captured, conv)()
                if isinstance(captured, dict):
                    event_id = result.get("event_id") or captured.get("event_id") or f"ANOM-{t}"
                    events.append({"event_id": event_id, "start_tick": t,
                                   "confirm_tick": t, "end_tick": None,
                                   "evidence": {**captured, "event_id": event_id},
                                   "values": [round(float(v), 5) for v in row]})

    notices = [f"Trained LSTM-AE (C68-Anomaly) at threshold {threshold:.3f} "
               f"({source})."]
    return {
        "scores": scores,
        "anomalous": flags,
        "events": events,
        "detector": {"backend": "external",
                     "name": "trained LSTM-AE (C68-Anomaly TEPApp)",
                     "threshold": round(threshold, 4),
                     "threshold_source": source,
                     "window": window, "stride": int(app.stride),
                     "confirm": 3, "calibration": source},
        "notices": notices,
    }


def status() -> Dict[str, Any]:
    repo = repo_dir()
    return {"repo": str(repo) if repo else None,
            "threshold": _STATE.get("threshold"),
            "threshold_source": _STATE.get("source")}
