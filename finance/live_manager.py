"""Live Real-Time Market Service, YTD Replay, and Persistent Portfolio Manager for Nifty 49 Equities.

Supports three distinct simulation modes:
  Mode 1: Final Benchmark Replay (Jan 2025 – Jan 2026) -> Replays primary test set via PaperEngine.
  Mode 2: 2026 YTD History Replay (Jan 1, 2026 – Yesterday) -> Replays the daily actions & rebalances that built the current portfolio.
  Mode 3: Live Real-Time Market Trading (Today's Live Session) -> Continues from yesterday's pre-existing portfolio,
          takes decisions in real time using live market data (Yahoo Finance), and settles into persistent storage
          so tomorrow continues seamlessly from today's closing state.
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import quote
from urllib.request import Request, urlopen
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

DEMO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_ROOT / 'data' / 'finance_paper'
STORE_PATH = DATA_DIR / 'live_portfolio_store.json'
BACKUP_PATH = DATA_DIR / 'live_portfolio_store.backup.json'
CACHE_PATH = DATA_DIR / 'live_market_cache.npz'

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

from finance.cash_execution import execute

TICKERS_49 = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS',
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS',
    'SHREECEM.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS',
    'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]

YF_TICKERS = [t.replace('LTIM.NS', 'LTM.NS') for t in TICKERS_49]


class LiveMarketService:
    """Fetches, caches, and provides current and real-time market data for all 49 Nifty equities."""

    @classmethod
    def get_market_status(cls) -> Tuple[str, bool]:
        """Determine whether the Indian equity market (NSE) is currently open."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_ist = now_utc + datetime.timedelta(hours=5, minutes=30)
        is_weekday = now_ist.weekday() < 5
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        curr_time = now_ist.time()

        if is_weekday and market_open <= curr_time <= market_close:
            return f"● LIVE MARKET OPEN (NSE {curr_time.strftime('%H:%M:%S')} IST)", True
        elif is_weekday and curr_time < market_open:
            return f"○ PRE-MARKET (NSE Opens at 09:15 IST · Currently {curr_time.strftime('%H:%M')} IST)", False
        else:
            return f"● MARKET CLOSED (Settled Daily Session · {now_ist.strftime('%Y-%m-%d %H:%M')} IST)", False

    @classmethod
    def fetch_latest_data(cls, start_date='2025-06-01'):
        """Download latest historical and current prices from Yahoo Finance."""
        try:
            import yfinance as yf
            end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            df = yf.download(YF_TICKERS, start=start_date, end=end_date, group_by='ticker', progress=False)
            if df.empty:
                return CACHE_PATH if CACHE_PATH.exists() else None

            dates = [d.strftime('%Y-%m-%d') for d in df.index]
            n_dates = len(dates)
            n_tickers = len(TICKERS_49)

            open_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
            close_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
            adj_close_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
            vol_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)

            for i, (t, yt) in enumerate(zip(TICKERS_49, YF_TICKERS)):
                open_arr[:, i] = df[yt]['Open'].to_numpy()
                close_arr[:, i] = df[yt]['Close'].to_numpy()
                adj_close_arr[:, i] = df[yt]['Close'].to_numpy()
                vol_arr[:, i] = df[yt]['Volume'].to_numpy()

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                CACHE_PATH,
                dates=np.array(dates),
                tickers=np.array(TICKERS_49),
                open=open_arr,
                close=close_arr,
                prices=adj_close_arr,
                volume=vol_arr
            )
            return CACHE_PATH
        except Exception as e:
            print(f"[LiveMarketService] Warning: Could not fetch online quotes ({e}). Using local cache.")
            return CACHE_PATH if CACHE_PATH.exists() else None

    @staticmethod
    def fetch_symbol(symbol):
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol, safe='') + '?range=6mo&interval=1d'
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=6) as response:
            result = json.load(response)['chart']['result'][0]
        meta = result['meta']
        price = float(meta['regularMarketPrice'])
        if meta['symbol'] != symbol or meta['currency'] != 'INR' or not np.isfinite(price) or price <= 0:
            raise ValueError('Invalid market quote for ' + symbol)
        closes = {datetime.datetime.fromtimestamp(ts, IST).date().isoformat(): float(value)
                  for ts, value in zip(result['timestamp'], result['indicators']['quote'][0]['close'])
                  if value is not None and np.isfinite(value) and value > 0}
        timestamp = int(meta['regularMarketTime'])
        date = datetime.datetime.fromtimestamp(timestamp, IST).date().isoformat()
        closes[date] = price
        return dict(price=price, timestamp=timestamp, date=date, closes=closes)

    @classmethod
    def fetch_realtime_quotes(cls):
        with ThreadPoolExecutor(max_workers=6) as pool:
            quotes = dict(zip(TICKERS_49, pool.map(cls.fetch_symbol, YF_TICKERS)))
        # Some daily series omit yesterday while today's quote is still provisional.
        if CACHE_PATH.exists():
            panel = cls.load_panel()
            for i, ticker in enumerate(panel['tickers']):
                entry = quotes[ticker]
                for date, price in zip(panel['dates'][-90:], panel['prices'][-90:, i]):
                    if date < entry['date'] and np.isfinite(price) and price > 0:
                        entry['closes'].setdefault(str(date), float(price))
        return quotes

    @classmethod
    def load_panel(cls):
        """Load the cached live market panel."""
        if not CACHE_PATH.exists():
            cls.fetch_latest_data()
        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"Market cache not found at {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return {
            'dates': list(data['dates']),
            'tickers': list(data['tickers']),
            'prices': data['prices'],
            'open': data['open'],
            'close': data['close'],
            'volume': data['volume']
        }


# ==============================================================================
# MODE 2: 2026 YTD HISTORY REPLAY (Jan 1, 2026 – Yesterday)
# ==============================================================================
class YTDReplaySession:
    """Replays the historical trading progression from Jan 1, 2026 up to yesterday's date.
    
    Lets the user inspect day-by-day and rebalance-by-rebalance how the portfolio was built
    from starting cash up to yesterday's settled close.
    """

    def __init__(self, key="ytd_history_session", capital=100000.0):
        self.key = key
        self.capital = float(capital)
        self.lock = threading.RLock()
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            self.full_store = json.load(f)

        self.tickers = TICKERS_49
        self.history = self.full_store.get("history", [])
        self.activities = self.full_store.get("activities", [])
        self.total_sessions = len(self.history)
        self.current_step = min(1, self.total_sessions - 1)

    def step(self, mode="day"):
        with self.lock:
            if self.current_step >= self.total_sessions - 1:
                return False
            if mode == "rebalance":
                curr_date = self.history[self.current_step]["date"]
                next_step = self.current_step + 1
                while next_step < self.total_sessions - 1:
                    step_date = self.history[next_step]["date"]
                    if any(a["execution_date"] == step_date for a in self.activities):
                        break
                    next_step += 1
                self.current_step = next_step
            else:
                self.current_step += 1
            return True

    def reset(self):
        with self.lock:
            self.current_step = 1
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            h_slice = self.history[:self.current_step + 1]
            latest_h = h_slice[-1] if h_slice else {'date': '2026-01-02', 'equity': self.capital, 'drawdown': 0.0}
            curr_date = latest_h['date']

            act_slice = [a for a in self.activities if a['execution_date'] <= curr_date]
            latest_act = act_slice[-1] if act_slice else None

            eq = latest_h['equity']
            weights = latest_act['target_weights'] if latest_act else [0.0] * len(self.tickers)
            cash_val = eq * max(0.0, 1.0 - sum(weights)) if latest_act else eq

            rets = [h_slice[i]['equity'] / h_slice[i - 1]['equity'] - 1.0 for i in range(1, len(h_slice))]
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
            max_dd = max([h['drawdown'] for h in h_slice]) if h_slice else 0.0

            holdings = []
            for i, (t, w) in enumerate(zip(self.tickers, weights)):
                holdings.append({
                    'ticker': t,
                    'value': float(w * eq),
                    'weight': float(w),
                    'cumulative_market_pnl': 0.0
                })

            finished = (self.current_step >= self.total_sessions - 1)
            status_desc = (
                "2026 YTD History Replay Complete · Current portfolio established as of " + curr_date +
                ". Switch to Mode 3 for Live Real-Time Market Trading."
                if finished else
                f"2026 YTD History Replay · Replaying Day {self.current_step + 1} of {self.total_sessions} ({curr_date})"
            )

            return {
                'session_id': self.key,
                'mode': 'ytd',
                'is_live': False,
                'date': curr_date,
                'initial_date': self.history[0]['date'] if self.history else '2026-01-02',
                'capital': self.capital,
                'equity': eq,
                'cash': cash_val,
                'net_return': eq / self.capital - 1.0,
                'fees': sum(a.get('fee', 0.0) for a in act_slice),
                'maximum_drawdown': max_dd,
                'sharpe': sharpe,
                'sessions': self.current_step + 1,
                'total_sessions': self.total_sessions,
                'finished': finished,
                'next_rebalance_in': (5 - (self.current_step + 1) % 5) % 5,
                'holdings': holdings,
                'history': h_slice,
                'activities': act_slice,
                'currency': 'INR',
                'model': 'InternVL3-2B-State-RL',
                'market_status': status_desc,
                'updated_at': self.full_store.get('updated_at')
            }


# ==============================================================================
# MODE 3: LIVE REAL-TIME MARKET TRADING (Today's Live Session)
# ==============================================================================
class LiveRealtimeSession:
    """Mark paper positions at timestamped quotes and settle one allocation per day."""

    def __init__(self, key="live_market_session", capital=100000.0):
        self.key = key
        self.capital = float(capital)
        self.lock = threading.RLock()
        self.tickers = TICKERS_49
        self.live_quotes = {}
        self.today_activity = None
        self.is_settled_today = False
        self._load_settled_portfolio()

    def _load_settled_portfolio(self):
        """Load baseline portfolio state settled as of yesterday's close."""
        self.ledger = Path(os.getenv('POLYVL_LIVE_ACCOUNT') or DATA_DIR / 'live_account.json')
        ledger = self.ledger
        today = datetime.datetime.now(IST).date()
        source = ledger if ledger.exists() else STORE_PATH
        state = json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
        self.account_note = ''
        if not state or state['asof_date'] > today.isoformat():
            self.account_note = 'Fresh cash account. The previous ledger is missing or future-dated.'
            prior = (today - datetime.timedelta(days=1)).isoformat()
            state = dict(asof_date=prior, initial_date=today.isoformat(), equity=self.capital,
                         capital=self.capital, cash_weight=1., weights=[0.] * len(self.tickers))
        self.settled_state = state
        self.capital = float(state['capital'])
        self.settled_date = state['asof_date']
        self.is_settled_today = self.settled_date == today.isoformat()
        self.settled_equity = self.live_equity = float(state['equity'])
        self.live_weights = np.array(state['weights'], dtype=float)
        self.live_cash = self.live_equity * state['cash_weight']
        stored_units = state.get("shares")
        self.units = np.array([stored_units[t] for t in self.tickers], dtype=np.int64) if stored_units is not None else None
        self.quote_error = 'Refresh quotes to load market data.'
        self.quote_date = ''
        self.quote_time = 0

    def refresh_quotes(self):
        with self.lock:
            try:
                quotes = LiveMarketService.fetch_realtime_quotes()
                dates = {q['date'] for q in quotes.values()}
                if len(dates) != 1:
                    raise ValueError('Constituents have different trading dates.')
                quote_date = dates.pop()
                if quote_date < self.settled_date:
                    raise ValueError('Quotes precede the settled account.')
                prices = np.array([quotes[t]['price'] for t in self.tickers])
                if self.units is None:
                    reference = self.settled_state.get('closing_prices', {})
                    units = np.zeros(len(prices), dtype=np.int64)
                    fractional_cash = 0.
                    for i, t in enumerate(self.tickers):
                        if self.live_weights[i]:
                            basis = reference.get(t) or quotes[t]['closes'].get(self.settled_date)
                            if not basis:
                                raise ValueError('Missing settlement price for ' + t)
                            value = self.live_equity * self.live_weights[i]
                            units[i] = int(np.floor(value / basis + 1e-12))
                            fractional_cash += value - units[i] * basis
                    self.units = units
                    self.live_cash += fractional_cash
                    if fractional_cash > .005:
                        self.account_note = f"Legacy fractional holdings converted to whole shares. INR {fractional_cash:,.2f} returned to cash at settlement prices."
                values = self.units * prices
                self.live_equity = float(self.live_cash + values.sum())
                self.live_weights = values / self.live_equity
                self.live_quotes = quotes
                self.quote_time = min(q['timestamp'] for q in quotes.values())
                self.quote_date = quote_date
                self.quote_error = ''
            except Exception as exc:
                self.quote_error = 'Quotes unavailable: ' + str(exc)
            return self.snapshot()

    def decision_blocker(self):
        today = datetime.datetime.now(IST).date().isoformat()
        if self.quote_error:
            return self.quote_error
        if self.quote_date != today or not LiveMarketService.get_market_status()[1]:
            return 'Trading paused outside the current NSE session.'
        if not -60 <= time.time() - self.quote_time <= 1200:
            return 'Quotes are older than 20 minutes. Refresh before trading.'
        if self.settled_date >= today:
            return 'Already settled for today.'
        return ''

    def execute_live_decision(self, service):
        from finance.policy import predict
        with self.lock:
            blocker = self.decision_blocker()
            if blocker:
                raise ValueError(blocker)
            if self.today_activity:
                return self.snapshot()
            dates = sorted(set.intersection(*(set(q['closes']) for q in self.live_quotes.values())))
            if len(dates) < 61 or dates[-1] != self.quote_date:
                raise ValueError('Insufficient aligned price history for 60-session signals.')
            prices = np.array([[self.live_quotes[t]['closes'][d] for t in self.tickers] for d in dates])
            panel = SimpleNamespace(prices=prices, dates=dates, tickers=self.tickers)
            acc = SimpleNamespace(panel=panel, index=len(dates), cash=self.live_cash / self.live_equity,
                                  equity=self.live_equity, weights=self.live_weights)
            with service.lease('internvl3_state_rl'):
                target_w, receipt = predict(service, account=acc, allocation_evidence=True)
            blocker = self.decision_blocker()
            if blocker:
                raise ValueError(blocker)
            old_eq = self.live_equity
            fee_bps = 10.
            budget = execute(self.live_weights, acc.cash, target_w, np.ones(len(self.tickers)), fee_bps=fee_bps)
            price = prices[-1]
            next_units = np.floor(old_eq * budget.post_fee_stock_notional / price + 1e-12).astype(np.int64)
            delta = next_units - self.units
            traded = delta * price
            gross_traded = float(np.abs(traded).sum())
            fee = gross_traded * fee_bps / 10000.
            cash = float(self.live_cash - traded.sum() - fee)
            if cash < -1e-7:
                raise ValueError('Insufficient cash for whole-share orders and fees.')
            cash = max(0., cash)
            values = next_units * price
            equity = float(cash + values.sum())
            rationales = receipt.get('rationales', []) if isinstance(receipt, dict) else []

            today_str = datetime.datetime.now(IST).date().isoformat()
            activity = {
                'id': len(self.settled_state.get('activities', [])),
                'execution_date': today_str,
                'signal_date': self.quote_date,
                'quote_time': self.quote_time,
                'previous_weights': self.live_weights.tolist(),
                'target_weights': target_w.tolist(),
                'pre_trade_equity': float(old_eq),
                'fee': fee,
                'turnover': gross_traded / old_eq,
                'whole_shares': True,
                'model_receipt': receipt,
                'rationales': rationales,
                'trades': [{'ticker': t, 'amount': float(amount), 'shares': int(abs(delta[i])),
                            'price': float(price[i]), 'executed_weight': float(values[i] / equity),
                            'previous_weight': float(self.live_weights[i]), 'target_weight': float(target_w[i])}
                           for i, (t, amount) in enumerate(zip(self.tickers, traded)) if delta[i] != 0]
            }

            self.today_activity = activity
            self.live_weights = values / equity
            self.live_equity = equity
            self.live_cash = cash
            self.units = next_units
            return self.snapshot()

    def settle_today(self, target_date=None):
        """Commit today's live session into the paper account ledger.
        
        Advances the permanent portfolio baseline so that tomorrow continues from this closing state.
        """
        with self.lock:
            today_str = datetime.datetime.now(IST).date().isoformat()
            if target_date and target_date != today_str:
                raise ValueError('Only the current trading date can be settled.')
            if self.settled_date == today_str:
                return self.snapshot()
            if (self.quote_error or self.quote_date != today_str or not self.today_activity
                    or self.today_activity['execution_date'] != today_str):
                raise ValueError('A decision and quotes from today are required for settlement.')
            history = list(self.settled_state.get('history', []))
            activities = list(self.settled_state.get('activities', []))
            returns = list(self.settled_state.get('returns', []))

            day_ret = (self.live_equity / max(self.settled_equity, 1.0)) - 1.0
            peak = max(self.settled_state.get('peak', self.capital), self.live_equity)
            drawdown = float(1.0 - self.live_equity / peak)

            history.append({
                'date': today_str,
                'equity': float(self.live_equity),
                'drawdown': drawdown
            })

            if self.today_activity:
                activities.append(self.today_activity)

            returns.append(float(day_ret))
            cash_w = float(self.live_cash / max(self.live_equity, 1.0))

            updated_state = {
                'capital': self.capital,
                'current_index': self.settled_state.get('current_index', 0) + 1,
                'start_index': self.settled_state.get('start_index', 0),
                'asof_date': today_str,
                'initial_date': self.settled_state.get('initial_date', '2026-01-02'),
                'equity': float(self.live_equity),
                'cash_weight': cash_w,
                'weights': self.live_weights.tolist(),
                'pnl': self.settled_state.get('pnl', [0.0]*len(self.tickers)),
                'fees': float(self.settled_state.get('fees', 0.0) + (self.today_activity.get('fee', 0.0) if self.today_activity else 0.0)),
                'peak': float(peak),
                'returns': returns,
                'history': history,
                'activities': activities,
                'ref_untrained': self.settled_state.get('ref_untrained', {}),
                'ref_buy_hold': self.settled_state.get('ref_buy_hold', {}),
                'sessions_count': len(history),
                'updated_at': datetime.datetime.now().isoformat()
            }

            updated_state['shares'] = dict(zip(self.tickers, self.units.tolist()))
            updated_state['closing_prices'] = {t: q['price'] for t, q in self.live_quotes.items()}
            ledger = self.ledger
            pending = ledger.with_suffix('.tmp')
            pending.write_text(json.dumps(updated_state, allow_nan=False), encoding='utf-8')
            os.replace(pending, ledger)

            self.settled_state = updated_state
            self.settled_date = today_str
            self.settled_equity = float(self.live_equity)
            self.is_settled_today = True

            print(f"[LiveRealtimeSession] Successfully settled session {today_str}. Total sessions: {len(history)}.")
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            s = self.settled_state
            _, is_open = LiveMarketService.get_market_status()
            today_str = datetime.datetime.now(IST).date().isoformat()

            holdings = []
            for i, (t, w) in enumerate(zip(self.tickers, self.live_weights)):
                q = self.live_quotes.get(t, {})
                holdings.append({
                    'ticker': t,
                    'value': float(w * self.live_equity),
                    'weight': float(w),
                    'price': q.get('price', 0.0),
                    'quote_timestamp': q.get('timestamp'),
                    'shares': int(self.units[i]) if self.units is not None else None
                })

            rets = s.get('returns', [])
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
            drawdowns = [h['drawdown'] for h in s.get('history', [])]
            max_dd = max(drawdowns) if drawdowns else 0.0

            activities = list(s.get('activities', []))
            if self.today_activity and self.today_activity not in activities:
                activities.append(self.today_activity)

            intraday_pnl = self.live_equity - self.settled_equity
            intraday_pct = (intraday_pnl / self.settled_equity) if self.settled_equity > 0 else 0.0

            settle_note = (
                "Today's session has been committed to permanent history."
                if self.is_settled_today else
                "Session active. Click 'Settle & Close Today' to lock today's run into history for tomorrow."
            )

            return {
                'session_id': self.key,
                'mode': 'live',
                'is_live': True,
                'date': today_str,
                'settled_date': self.settled_date,
                'initial_date': s.get('initial_date', '2026-01-02'),
                'capital': self.capital,
                'equity': self.live_equity,
                'settled_equity': self.settled_equity,
                'intraday_pnl': intraday_pnl,
                'intraday_pct': intraday_pct,
                'cash': self.live_cash,
                'net_return': self.live_equity / self.capital - 1.0,
                'fees': s.get('fees', 0.0) + (self.today_activity['fee'] if self.today_activity and not self.is_settled_today else 0.),
                'maximum_drawdown': max_dd,
                'sharpe': sharpe,
                'sessions': len(s.get('history', [])),
                'total_sessions': len(s.get('history', [])),
                'finished': False,
                'next_rebalance_in': 0,
                'holdings': holdings,
                'history': s.get('history', []),
                'activities': activities,
                'currency': 'INR',
                'model': 'InternVL3-2B-State-RL',
                'market_status': (self.decision_blocker() or 'Market quotes available') + (' | Oldest quote ' + datetime.datetime.fromtimestamp(self.quote_time, IST).strftime('%d %b %H:%M IST') if self.quote_time else ''),
                'account_note': self.account_note,
                'can_decide': not self.decision_blocker() and not self.today_activity,
                'can_settle': bool(self.today_activity and self.today_activity['execution_date'] == today_str and self.quote_date == today_str and not self.quote_error and self.settled_date < today_str),
                'is_market_open': bool(is_open and not self.quote_error and self.quote_date == today_str and 0 <= time.time() - self.quote_time <= 1200),
                'is_settled_today': self.is_settled_today,
                'settle_note': settle_note,
                'updated_at': datetime.datetime.now().isoformat()
            }


# Global Singletons
_ytd_session: Optional[YTDReplaySession] = None
_live_session: Optional[LiveRealtimeSession] = None
_manager_lock = threading.Lock()

def get_ytd_session(capital=100000.0) -> YTDReplaySession:
    global _ytd_session
    with _manager_lock:
        if _ytd_session is None or _ytd_session.capital != capital:
            _ytd_session = YTDReplaySession(capital=capital)
        return _ytd_session

def get_live_realtime_session(capital=100000.0) -> LiveRealtimeSession:
    global _live_session
    with _manager_lock:
        if _live_session is None or (_live_session.today_activity and _live_session.today_activity['execution_date'] != datetime.datetime.now(IST).date().isoformat()):
            _live_session = LiveRealtimeSession(capital=capital)
        return _live_session

# Backward compatibility aliases
LivePortfolioManager = LiveRealtimeSession
def get_live_manager(capital=100000.0):
    return get_live_realtime_session(capital=capital)
