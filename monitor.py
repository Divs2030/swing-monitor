# monitor.py
# Monitors a fixed list of stocks for weekly RSI trigger -> breakout entries and SMA exits.
# Sends Telegram messages on Entry / Exit. Built for GitHub Actions scheduled runs (every 15 min).

import os
import json
import time
import requests
from datetime import datetime, time as dt_time, timedelta
import pytz
import pandas as pd
import numpy as np
import yfinance as yf

# ------------------ USER CONFIG (edit only stocks_list.txt; this file reads that list) ------------------
PER_TRADE_CAPITAL = float(os.getenv("PER_TRADE_CAPITAL", "30000"))  # e.g. 30000 rupees
EXIT_MODE = os.getenv("EXIT_MODE", "SMA36")  # 'SMA21' or 'SMA36' or 'BOTH'
RSI_PERIOD = 14
RSI_PAST_LOOKBACK = 52
RSI_PAST_THRESH = 56
RSI_NOW_LOW = 60
RSI_NOW_HIGH = 65
SMA21 = 21
SMA36 = 36

# State file in repo
STATE_FILE = "state.json"

# Telegram (set via GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")

# Timezone: India
IST = pytz.timezone("Asia/Kolkata")

# Market hours
MARKET_OPEN = dt_time(hour=9, minute=15)
MARKET_CLOSE = dt_time(hour=15, minute=30)

# ------------------ Helpers ------------------
def load_stocks(filename="stocks_list.txt"):
    with open(filename, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    return symbols

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, default=str, indent=2)

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Message:", text)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print("Telegram send error:", e)
        return False

# RSI computation (pandas Series)
def compute_rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period-1, adjust=True, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def is_market_time(now_ist):
    # Monday=0 ... Sunday=6
    weekday = now_ist.weekday()
    if weekday >= 5:
        return False
    t = now_ist.time()
    return (t >= MARKET_OPEN) and (t <= MARKET_CLOSE)

def friday_after_close(now_ist):
    # Friday after market close (we'll use >= 15:40 Fri IST)
    return (now_ist.weekday() == 4) and (now_ist.time() >= (dt_time(hour=15, minute=40)))

def resample_to_weekly(df_daily):
    df = df_daily.copy()
    df.index = pd.to_datetime(df.index)
    weekly = pd.DataFrame()
    weekly['Open'] = df['Open'].resample('W-FRI').first()
    weekly['High'] = df['High'].resample('W-FRI').max()
    weekly['Low'] = df['Low'].resample('W-FRI').min()
    weekly['Close'] = df['Close'].resample('W-FRI').last()
    weekly['Volume'] = df['Volume'].resample('W-FRI').sum()
    weekly.dropna(inplace=True)
    return weekly

# ------------------ Core logic ------------------
def compute_weekly_triggers(symbols, state):
    """
    For each symbol compute weekly RSI and detect trigger candles (strict cross-up).
    If trigger found and not already active, record trigger_high and trigger_date in state.
    """
    print("Computing weekly triggers for", len(symbols), "symbols")
    for sym in symbols:
        try:
            yahoo_sym = sym if ('.' in sym) else sym + ".NS"
            df = yf.download(yahoo_sym, period="5y", interval="1d", progress=False, threads=False)
            if df.empty:
                # try BSE
                yahoo_sym = sym + ".BO"
                df = yf.download(yahoo_sym, period="5y", interval="1d", progress=False, threads=False)
            if df.empty:
                print("No daily data for", sym)
                continue
            weekly = resample_to_weekly(df)
            if weekly.empty or len(weekly) < RSI_PERIOD + 2:
                continue
            weekly['RSI'] = compute_rsi(weekly['Close'], RSI_PERIOD)
            # strict cross-up: prev < RSI_NOW_LOW and now >= RSI_NOW_LOW and now < RSI_NOW_HIGH
            for i in range(1, len(weekly)):
                now_rsi = weekly['RSI'].iloc[i]
                prev_rsi = weekly['RSI'].iloc[i-1]
                # past low check
                start_look = max(0, i - RSI_PAST_LOOKBACK)
                past_min = weekly['RSI'].iloc[start_look:i].min() if i > start_look else np.nan
                if pd.isna(now_rsi) or pd.isna(prev_rsi) or pd.isna(past_min):
                    continue
                if (past_min < RSI_PAST_THRESH) and (prev_rsi < RSI_NOW_LOW) and (now_rsi >= RSI_NOW_LOW) and (now_rsi < RSI_NOW_HIGH):
                    trigger_high = float(weekly['High'].iloc[i])
                    trigger_date = str(weekly.index[i].date())
                    # add to state only if not active already and not same trigger
                    s = state.get(sym, {})
                    # If there is an active trade for sym, skip (we don't want overlapping)
                    if s.get('active'):
                        break
                    # Only set trigger if not already set for same trigger_date
                    if s.get('trigger_date') == trigger_date and s.get('trigger_high') == trigger_high:
                        break
                    # Set trigger in state
                    state[sym] = {
                        "trigger_date": trigger_date,
                        "trigger_high": trigger_high,
                        "trigger_computed_on": str(datetime.now(IST)),
                        "active": False
                    }
                    print(f"Trigger set for {sym}: {trigger_date} @ {trigger_high}")
                    break
        except Exception as e:
            print("Error computing weekly trigger for", sym, e)
    return state

def check_intraday_breakouts(symbols, state):
    """
    Poll intraday 15m bars for today and check if any bar's High > trigger_high.
    If breakout and symbol not active, mark active and send telegram entry.
    """
    print("Checking intraday breakouts (batch download). Symbols:", len(symbols))
    # Use batch download of 1d 15m bars
    tickers = []
    for s in symbols:
        yahoo = s if ('.' in s) else s + ".NS"
        tickers.append(yahoo)
    try:
        data = yf.download(tickers=tickers, period="1d", interval="15m", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print("Intraday download error:", e)
        return state

    for s in symbols:
        try:
            s_y = s if ('.' in s) else s + ".NS"
            # fallback if key missing
            if (s_y not in data) and (s + ".BO" in data):
                s_y = s + ".BO"
            if s_y not in data:
                print("No intraday data for", s)
                continue
            df_intraday = data[s_y].dropna()
            if df_intraday.empty:
                continue
            # get max High for today so far
            todays_high = float(df_intraday['High'].max())
            st = state.get(s, {})
            trig = st.get('trigger_high')
            if trig is None:
                continue
            if st.get('active'):
                continue  # already active
            # breakout condition
            if todays_high > float(trig):
                # mark entry
                entry_price = float(trig)  # we notify at trigger_high
                qty = int(np.floor(PER_TRADE_CAPITAL / entry_price))
                if qty <= 0:
                    message = f"ENTRY (SKIPPED: insufficient qty) — {s}\nTrigger: {st.get('trigger_date')} @ {trig}\nBreakout detected (today high {todays_high})\nPer-trade capital {PER_TRADE_CAPITAL} → qty=0"
                    send_telegram(message)
                else:
                    # mark active and record entry_date
                    st.update({
                        "active": True,
                        "entry_date": str(datetime.now(IST)),
                        "entry_price": entry_price,
                        "qty": qty,
                        "exit_method": EXIT_MODE
                    })
                    state[s] = st
                    message = (f"ENTRY — {s}\nTrigger date: {st.get('trigger_date')} — trigger_high: {trig}\n"
                               f"Breakout detected (today high {todays_high})\nEntry price: {entry_price}\nQty: {qty}\nExit rule: {EXIT_MODE}")
                    send_telegram(message)
                    print("Entry sent for", s)
        except Exception as e:
                    print("Error in compute_weekly_triggers:", e)
                    # optionally continue to the next symbol or return partial state
        return state

def check_weekly_exits(symbols, state):
    """
    Friday after close: compute weekly closes and SMA; if weekly close < SMA (exit rule), then mark exit and notify.
    """
    print("Checking weekly exits for", len(symbols))
    for s in symbols:
        try:
            yahoo_sym = s if ('.' in s) else s + ".NS"
            df = yf.download(yahoo_sym, period="2y", interval="1d", progress=False, threads=False)
            if df.empty:
                yahoo_sym = s + ".BO"
                df = yf.download(yahoo_sym, period="2y", interval="1d", progress=False, threads=False)
            if df.empty:
                continue
            weekly = resample_to_weekly(df)
            if weekly.empty:
                continue
            # last weekly close and SMA
            weekly['SMA21'] = weekly['Close'].rolling(SMA21).mean()
            weekly['SMA36'] = weekly['Close'].rolling(SMA36).mean()
            last_close = float(weekly['Close'].iloc[-1])
            sma21 = float(weekly['SMA21'].iloc[-1]) if not pd.isna(weekly['SMA21'].iloc[-1]) else None
            sma36 = float(weekly['SMA36'].iloc[-1]) if not pd.isna(weekly['SMA36'].iloc[-1]) else None

            st = state.get(s, {})
            if not st.get('active'):
                # nothing to exit
                continue

            exit_hit = False
            exit_price = None
            method = st.get('exit_method', EXIT_MODE)
            if method == "SMA21" or method == "BOTH":
                if sma21 is not None and last_close < sma21:
                    exit_hit = True
                    exit_price = last_close
                    method = "SMA21"
            if not exit_hit and method in ("SMA36", "BOTH"):
                if sma36 is not None and last_close < sma36:
                    exit_hit = True
                    exit_price = last_close
                    method = "SMA36"

            if exit_hit:
                qty = st.get('qty', 0)
                entry_price = st.get('entry_price', None)
                pnl = None
                if entry_price is not None and qty > 0:
                    pnl = (exit_price - entry_price) * qty
                # record exit in state history
                history = st.get('history', [])
                history.append({
                    "entry_date": st.get('entry_date'),
                    "entry_price": st.get('entry_price'),
                    "exit_date": str(datetime.now(IST)),
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_method": method
                })
                # clear active flag and trigger (so future triggers can be set)
                new_st = {
                    "trigger_date": None,
                    "trigger_high": None,
                    "trigger_computed_on": None,
                    "active": False,
                    "history": history
                }
                state[s] = new_st
                msg = (f"EXIT — {s}\nExit method: {method}\nExit price (weekly close): {exit_price}\nQty: {qty}\n"
                       f"P&L (gross): {pnl}\nEntry was: {entry_price}\n")
                send_telegram(msg)
                print("Exit sent for", s)
        except Exception as e:
            print("Error checking exit for", s, e)
    return state

# ------------------ Main entrypoint ------------------
def main():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    now_ist = now_utc.astimezone(IST)
    print("Now IST:", now_ist)

    symbols = load_stocks()
    state = load_state()

    # 1) If Friday after close -> compute triggers and also check weekly exits
if friday_after_close(now_ist):
    print("Friday after close: recomputing weekly triggers and checking exits")

    state_before = json.dumps(state, sort_keys=True)

    state = compute_weekly_triggers(symbols, state)
    state = check_weekly_exits(symbols, state)
    save_state(state)

    state_after = json.dumps(state, sort_keys=True)

    if state_before == state_after:
        send_telegram(
            "🟢 Weekly Scan Completed\n"
            "Day: Friday\n"
            "Result: No new entries or exits found."
        )
    else:
        send_telegram(
            "🟢 Weekly Scan Completed\n"
            "Day: Friday\n"
            "Result: State updated (new trigger / entry / exit)."
        )
    return

    # 2) If during market hours -> poll intraday and check breakouts
    if is_market_time(now_ist):
        print("Market hours: performing intraday poll and checking breakouts")
        state = check_intraday_breakouts(symbols, state)
        save_state(state)
        return

    # 3) Outside market hours -> do nothing (or optionally compute triggers if you want)
    print("Outside market hours - nothing to do. Next run will act as scheduled.")
    return

if __name__ == "__main__":
    # Test signal when workflow is manually triggered
    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram("✅ System Test Successful\nSwing monitor is running correctly.")
    main()








