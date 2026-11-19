import datetime
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def live_account(tmp_path, monkeypatch):
    from api import server
    from finance import live_manager as live, policy

    monkeypatch.setattr(live, '_live_session', None)
    monkeypatch.setattr(live, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(live, 'STORE_PATH', tmp_path / 'legacy.json')
    monkeypatch.delenv('POLYVL_LIVE_ACCOUNT', raising=False)
    today = datetime.datetime.now(live.IST).date()
    dates = [(today - datetime.timedelta(days=n)).isoformat() for n in range(70, -1, -1)]
    quotes = {ticker: dict(date=dates[-1], timestamp=int(time.time()), price=170.,
                           closes={d: 100. + i for i, d in enumerate(dates)})
              for ticker in live.TICKERS_49}
    monkeypatch.setattr(live.LiveMarketService, 'fetch_realtime_quotes', lambda: quotes)
    monkeypatch.setattr(live.LiveMarketService, 'get_market_status', lambda: ('Open', True))
    monkeypatch.setattr(server, 'get_service', lambda: SimpleNamespace(lease=lambda task: nullcontext()))

    def predict(service, account, *, allocation_evidence):
        assert allocation_evidence
        _, _, _, features = policy.build_state_prompt(account)
        assert features[live.TICKERS_49[0]]['price'] == quotes[live.TICKERS_49[0]]['price']
        weights = np.zeros(49)
        weights[:10] = .1
        return weights, {'mode': 'internvl3_state_rl'}

    monkeypatch.setattr(policy, 'predict', predict)
    return quotes
