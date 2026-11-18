import json
import pytest
from fastapi.testclient import TestClient
from api.server import app
from tep_sim.sensors import FAULTS

client = TestClient(app)

def test_anomaly_scenarios_categorized_normals_and_faults():
    r = client.get("/anomaly/scenarios")
    assert r.status_code == 200
    data = r.json()
    scenarios = data["scenarios"]
    assert len(scenarios) > 0

    normals = [s for s in scenarios if s.get("category") == "normal"]
    faults = [s for s in scenarios if s.get("category") == "fault"]

    assert len(normals) >= 3, "Expected at least 3 normal run scenarios"
    assert len(faults) >= 20, "Expected at least 20 fault run scenarios"

    # Verify normal runs are fault 0 and have clean blind labels
    for s in normals:
        assert s["fault"] == 0
        assert "Normal" in s["blind_label"]
        assert s["category"] == "normal"

    # Verify fault runs are fault > 0 and blind labels do not spoil physical fault description
    for s in faults:
        assert s["fault"] > 0
        assert "Fault" in s["blind_label"]
        assert s["category"] == "fault"


def test_anomaly_run_model_classification():
    r = client.get("/anomaly/scenarios")
    data = r.json()
    # Pick a fault scenario e.g. Fault 4 or Fault 1
    faulty = next((s for s in data["scenarios"] if s["fault"] == 4), None)
    if faulty is None:
        faulty = next(s for s in data["scenarios"] if s["fault"])

    run_res = client.post("/anomaly/run", json={"scenario_id": faulty["id"]})
    assert run_res.status_code == 200
    res_data = run_res.json()
    assert "events" in res_data
    assert len(res_data["events"]) > 0

    ev = res_data["events"][0]
    assert "diagnosis" in ev
    diag = ev["diagnosis"]
    assert diag is not None
    assert "predicted_fault" in diag or "predicted_class" in diag
    assert "confidence" in diag


@pytest.mark.parametrize("condition", ["mixed", "normal", "fault"])
def test_draw_run_by_condition(condition, monkeypatch):
    from tep_sim import detector
    monkeypatch.setattr(detector, "_external", lambda *args: None)
    response = client.post("/anomaly/run", json={"condition": condition})
    assert response.status_code == 200
    result = response.json()
    assert result["scenario"]["category"] in (
        {"normal", "fault"} if condition == "mixed" else {condition})
    for event in result["events"]:
        assert event["event_id"] == f"ANOM-{event['confirm_tick']:04d}"


def test_trained_stream_receives_only_measurements_and_index(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from tep_sim import trained

    records = []
    app = SimpleNamespace(window_size=1, stride=1, process_sensor_stream=records.append)
    monkeypatch.setattr(trained, "repo_dir", lambda: tmp_path)
    monkeypatch.setattr(trained, "_build", lambda *args: (app, 1., "test", None))
    trained.replay(np.zeros((2, 52)))
    assert records == [{"sample_index": i, "values": [0.] * 52} for i in range(2)]


def test_abstention_reaches_reasoner_without_promoting_ranked_fault():
    from tep_sim.detector import telemetry_block
    from tep_sim.trained import _diagnosis_of
    diagnosis = _diagnosis_of({"report": {"diagnosis": {
        "fault_id": "UNCLASSIFIED_OPEN_SET", "predicted_class": "FAULT_04",
        "verdict": "unclassified", "abstained": True}}})
    assert diagnosis["predicted_fault"] is None
    block = telemetry_block({"diagnosis": diagnosis})
    assert "UNCLASSIFIED_OPEN_SET" in block
    assert "FAULT_04" not in block


@pytest.mark.parametrize("flush_at_end", [False, True])
def test_replay_preserves_alarm_and_confirmation_between_windows(monkeypatch, tmp_path, flush_at_end):
    from types import SimpleNamespace
    import numpy as np
    from tep_sim import trained

    payload = {"event_id": "ANOM-1", "evidence": {}}
    def process(record):
        t = record["sample_index"]
        scored = t in (1, 3, 5, 7)
        closes = t == 7 and not flush_at_end
        return {"window_score": (0.2 if closes else 2.0) if scored else None,
                "is_anomalous": scored and not closes,
                "open_event": t >= 5 and not closes and (t < 7 or flush_at_end),
                "anomaly_detected": closes, "event": payload if closes else None}

    app = SimpleNamespace(window_size=2, stride=2, process_sensor_stream=process,
                          finalize_stream=lambda: payload if flush_at_end else None)
    monkeypatch.setattr(trained, "repo_dir", lambda: tmp_path)
    monkeypatch.setattr(trained, "_build", lambda *args: (app, 1., "test", None))
    timeline = trained.replay(np.zeros((8, 52)))
    assert timeline["anomalous"] == [0, 1, 1, 1, 1, 1, 1, int(flush_at_end)]
    event, = timeline["events"]
    assert (event["start_tick"], event["confirm_tick"], event["report_tick"]) == (1, 5, 7)
    assert event["end_tick"] == (None if flush_at_end else 7)
