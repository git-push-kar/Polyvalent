"""Tests for Live Real-Time Market Simulation Mode and Persistent Storage."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from api.server import app
from finance.live_manager import get_live_manager


def test_live_manager_snapshot(live_account):
    """Verify LivePortfolioManager loads persistent state and produces a valid snapshot."""
    mgr = get_live_manager()
    snap = mgr.snapshot()

    assert snap['mode'] == 'live'
    assert snap['is_live'] is True
    assert snap['initial_date'] == snap['date']
    assert snap['capital'] == 100000.0
    assert snap['equity'] > 0
    assert len(snap['holdings']) == 49
    assert snap['cash'] == snap['equity']
    assert not snap['can_decide']


def test_live_session_api_endpoints(live_account):
    """Verify API endpoints for live session creation, snapshot, and refresh."""
    client = TestClient(app)

    # 1. Start live session
    res = client.post('/finance/paper/sessions', json={'capital': 100000.0, 'mode': 'live'})
    assert res.status_code == 200
    data = res.json()
    assert data['session_id'] == 'live_market_session'
    assert data['is_live'] is True
    assert len(data['holdings']) == 49

    # 2. Get live snapshot
    res_snap = client.get('/finance/paper/sessions/live_market_session')
    assert res_snap.status_code == 200
    snap = res_snap.json()
    assert snap['session_id'] == 'live_market_session'
    assert snap['equity'] == data['equity']

    # 3. Live refresh endpoint
    res_ref = client.post('/finance/paper/live/refresh')
    assert res_ref.status_code == 200
    ref_data = res_ref.json()
    assert ref_data['session_id'] == 'live_market_session'
    assert ref_data['is_live'] is True


def test_stale_quote_and_model_failure_leave_cash_intact(live_account, monkeypatch):
    from finance import policy

    client = TestClient(app)
    client.post('/finance/paper/sessions', json={'mode': 'live'})
    def fail(*args, **kwargs):
        raise RuntimeError('test model failure')
    monkeypatch.setattr(policy, 'predict', fail)
    assert client.post('/finance/paper/live/decide').status_code == 503
    for q in live_account.values():
        q['timestamp'] -= 3600
    snapshot = client.post('/finance/paper/live/refresh').json()
    assert not snapshot['can_decide']
    assert client.post('/finance/paper/live/decide').status_code == 409
    assert snapshot['equity'] == snapshot['cash'] == 100000.
    assert not snapshot['activities']


def test_quotes_revalue_existing_shares_after_rebalance(live_account):
    client = TestClient(app)
    client.post('/finance/paper/sessions', json={'mode': 'live'})
    before = client.post('/finance/paper/live/decide').json()
    position = next(h for h in before['holdings'] if h['value'] > 0)
    quote = live_account[position['ticker']]
    quote['price'] *= 1.05
    quote['closes'][quote['date']] = quote['price']
    after = client.post('/finance/paper/live/refresh').json()
    assert after['cash'] == before['cash']
    assert after['equity'] == pytest.approx(before['equity'] + position['value'] * .05)
    assert client.post('/finance/paper/live/refresh').json()['equity'] == after['equity']


def test_whole_share_fills_and_legacy_conversion(live_account, tmp_path):
    import json
    from finance import live_manager as live

    today = next(iter(live_account.values()))['date']
    previous = sorted(next(iter(live_account.values()))['closes'])[-2]
    units = 2.5
    basis = next(iter(live_account.values()))['closes'][previous]
    equity = units * basis
    live.STORE_PATH.write_text(json.dumps(dict(asof_date=previous, capital=equity,
        equity=equity, cash_weight=0., weights=[1.] + [0.]*48)))
    client = TestClient(app)
    initial = client.post('/finance/paper/sessions', json={'mode':'live'}).json()
    assert initial['cash'] == pytest.approx(.5 * basis)
    assert initial['holdings'][0]['shares'] == 2
    state = client.post('/finance/paper/live/decide').json()
    fills = state['activities'][-1]['trades']
    assert len(fills) == 1 and fills[0]['shares'] == 2
    assert fills[0]['amount'] == -2 * live_account[live.TICKERS_49[0]]['price']
    assert all(isinstance(h['shares'], int) and h['shares'] == 0 for h in state['holdings'])
    assert state['cash'] == pytest.approx(initial['cash'] - fills[0]['amount'] - state['fees'])
    assert client.post('/finance/paper/live/settle').json()['settled_date'] == today
    saved = json.loads((tmp_path / 'live_account.json').read_text())
    assert list(saved['shares'].values()) == [0]*49
