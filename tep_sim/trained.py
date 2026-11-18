"""Bridge to the trained TEP stack of the `anomaly/` training repository.

The deployment artifacts are split in two halves (see
`anomaly/docs/reports/HYBRID_LIVE_INTEGRATION_PLAN_2026-11-16.md` §9):

  L1 diagnosis — `outputs/analysis/phase3a/hybrid/vocabfix/protocolA/`
                 LightGBM booster over 323 page features, isotonic calibration,
                 menu rule over 15 faults + `UNCLASSIFIED_OPEN_SET`. CPU only.
  L2 writer    — `outputs/tep_rca_internvl3_2b` (InternVL3-2B + LoRA), used by
                 `adapters/anomaly` in this repo — byte-identical to the
                 deployment weights (SHA-256 050ee467…).

`TEPApp` already owns the seam: with `config["hybrid"]["enabled"]` it renders the
classifier page from each closed event's evidence, scores it and stores the
verdict as `report["diagnosis"]`, never touching the prose. This module turns
that on, replays a run, closes the trailing event so its verdict is not lost,
and reports both halves.

Two ports are supported: the repository holding the L1 artifacts is preferred,
and `C68-Anomaly` (detector pipeline only, no L1) still works as a fallback.
That fallback needs two repairs, applied here instead of editing it:

  * its `config.json` claims hidden 128 / latent 64 while `model.pt` is 32/16,
  * its `threshold.json` (0.687) belongs to a larger model, so the shipped net
    alarms on every fault-free window.

Shape inference always runs (harmless when the metadata is right); the operating
point is only re-derived when the stored threshold actually fails the fault-free
runs, which keeps the training repo's calibrated value intact.
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

from .dataset import Scenario, dataset_dir, load

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_L1_ARTIFACTS = Path("outputs/analysis/phase3a/hybrid/vocabfix/protocolA/classifier.txt")
_PERCENTILE = 99.0
_FALSE_ALARM_GUARD = 0.20   # stored threshold failing this share of normal windows
_STATE: Dict[str, Any] = {"threshold": None, "source": None, "repo": None,
                          "hybrid": None}
_LOCK = threading.Lock()


def repo_dir() -> Optional[Path]:
    """Locate the training checkout, preferring the one that carries L1."""
    raw = (os.getenv("FINAL_ANOMALY_REPO") or "").strip().strip('"')
    candidates: List[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates += [_REPO.parent / "anomaly", _REPO.parent / "C68-Anomaly"]

    usable: List[Path] = []
    for cand in candidates:
        if not cand.is_absolute():
            cand = _REPO / cand
        if (cand / "main.py").is_file() and (cand / "anomaly_detection").is_dir():
            usable.append(cand.resolve())
    for cand in usable:                     # a checkout with the L1 booster wins
        if (cand / _L1_ARTIFACTS).is_file():
            return cand
    return usable[0] if usable else None


# -- loader repair (C68-Anomaly only, harmless elsewhere) --------------------

def _patch_loader() -> None:
    """Make `AnomalyDetector.from_artifacts` infer the architecture from tensors."""
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

    stored = float(app.detector.threshold.threshold)
    window, stride = int(app.window_size), int(app.stride)
    scores: List[float] = []
    for run in _fault_free_runs():
        for t in range(0, len(run) - window + 1, stride):
            score, _ = app.detector.score_window(run[t:t + window])
            scores.append(float(score))
    if not scores:
        return stored, "threshold.json (no fault-free runs to check against)"

    values = np.asarray(scores)
    false_rate = float((values > stored).mean())
    if false_rate <= _FALSE_ALARM_GUARD:
        return stored, (f"threshold.json (alarms {false_rate:.1%} of "
                        f"{len(values)} fault-free windows)")
    derived = float(np.percentile(values, _PERCENTILE))
    logger.warning("stored threshold %.3f alarms %.0f%% of fault-free windows; "
                   "re-deriving at p%g = %.3f",
                   stored, false_rate * 100, _PERCENTILE, derived)
    return derived, (f"re-derived p{_PERCENTILE:g} of {len(values)} fault-free "
                     f"windows (threshold.json {stored:.3f} alarmed "
                     f"{false_rate:.0%})")


def _build(repo: Path) -> Tuple[Any, float, str, Optional[str]]:
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
    # This test export injects faults at sample 20 (data/tep_test/dataset.json),
    # while the training config assumes 160 for the 48 h Braatz runs. The onset
    # drives the pre-fault context the L1 page renders, so a stale value shifts
    # every page the classifier reads. Follow the dataset that is replayed.
    from .dataset import fault_onset

    onset = int(fault_onset())
    dataset_cfg = config.setdefault("dataset", {})
    fault_onsets = dataset_cfg.setdefault("fault_onset", {})
    fault_onsets["faulty_testing"] = onset
    fault_onsets["faulty_training"] = onset
    dataset_cfg["fault_onset_index"] = onset
    # L1 diagnosis (§9.5): the verdict is its own field, the prose is untouched.
    hybrid_cfg = config.setdefault("hybrid", {})
    hybrid_cfg["enabled"] = True
    hybrid_cfg.setdefault("artifact_dir",
                          str(Path("outputs/analysis/phase3a/hybrid/vocabfix")))
    # The menu threshold is a pre-registered false-abstention budget, not a knob
    # (§9.9); the stored 1.0 is used unless a deployment pins another value.
    menu_threshold = (os.getenv("FINAL_ANOMALY_MENU_THRESHOLD") or "").strip()
    if menu_threshold:
        hybrid_cfg["threshold"] = float(menu_threshold)
    with contextlib.chdir(repo):
        app = TEPApp(config, enable_llm=False)
        threshold, source = _operating_point(app, repo)
    app.detector.threshold.threshold = float(threshold)
    hybrid_error = None if getattr(app, "hybrid", None) is not None else "hybrid unavailable"
    return app, threshold, source, hybrid_error


# -- replay ------------------------------------------------------------------

def _diagnosis_of(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    report = event.get("report") or {}
    verdict = report.get("diagnosis") if isinstance(report, dict) else None
    return verdict if isinstance(verdict, dict) else None


def _as_event(captured: Any, tick: int, row: np.ndarray,
              event_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for conv in ("model_dump", "dict", "to_dict"):
        if not isinstance(captured, dict) and hasattr(captured, conv):
            captured = getattr(captured, conv)()
    if not isinstance(captured, dict):
        return None
    ident = event_id or captured.get("event_id") or f"ANOM-{tick}"
    evidence = captured.get("evidence") if isinstance(captured.get("evidence"), dict) else captured
    return {"event_id": ident, "start_tick": tick, "confirm_tick": tick,
            "end_tick": captured.get("end_sample") or None,
            "evidence": {**evidence, "event_id": ident},
            "diagnosis": _diagnosis_of(captured),
            "event": captured,
            "values": [round(float(v), 5) for v in row]}


def replay(scenario: Scenario, samples: Optional[np.ndarray] = None
           ) -> Dict[str, Any]:
    """Replay one run through the trained pipeline (fresh state per run)."""
    repo = repo_dir()
    if repo is None:
        raise FileNotFoundError("no anomaly training checkout found")

    with _LOCK:
        app, threshold, source, hybrid_error = _build(repo)
        _STATE.update({"threshold": threshold, "source": source,
                       "repo": str(repo), "hybrid": hybrid_error})

    x = load(scenario) if samples is None else np.asarray(samples)
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
                built = _as_event(result["event"], t, row, result.get("event_id"))
                if built:
                    events.append(built)
        # A finite replay usually ends while the event is still open; closing it
        # is what produces the report (and therefore the L1 verdict).
        tail = app.finalize_stream() if hasattr(app, "finalize_stream") else None
        if tail:
            built = _as_event(tail, len(x) - 1, x[-1])
            if built and built["event_id"] not in {e["event_id"] for e in events}:
                events.append(built)

    diagnoses = [e["diagnosis"] for e in events if e.get("diagnosis")]
    notices = [f"Trained LSTM-AE + hybrid L1 at threshold {threshold:.3f} ({source})."]
    if hybrid_error:
        notices.append("L1 classifier unavailable in this checkout; "
                       "events carry no diagnosis.")
    detector = {"backend": "external",
                "name": "trained LSTM-AE + CVA (TEPApp)",
                "threshold": round(threshold, 4),
                "threshold_source": source,
                "window": int(app.window_size), "stride": int(app.stride),
                "confirm": 3, "calibration": source,
                "diagnosis": "classifier+menu_rule" if not hybrid_error else None,
                "diagnosis_source": str(getattr(getattr(app, "hybrid", None),
                                                "artifacts", "") or "") or None,
                "repo": str(repo)}
    return {
        "scores": scores,
        "anomalous": flags,
        "events": events,
        "diagnosis": diagnoses[-1] if diagnoses else None,
        "diagnoses": diagnoses,
        "detector": detector,
        "notices": notices,
    }


def status() -> Dict[str, Any]:
    repo = repo_dir()
    return {"repo": str(repo) if repo else None,
            "threshold": _STATE.get("threshold"),
            "threshold_source": _STATE.get("source"),
            "hybrid_error": _STATE.get("hybrid")}
