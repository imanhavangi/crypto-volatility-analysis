#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live 1m signal runner for two models (separate balances, shared Telegram channel).
- Every minute: fetch last 60 candles
- Build same features as training (via OptimalTradingAnalyzer.calculate_technical_indicators)
- Run both models (fold1 & fold2), apply calibrated tau & gamma
- If entry -> open position with TP & SL; track PnL and close on TP/SL
- Send all reports to one Telegram channel with model tag
"""

import os
import time
import json
import math
import signal
import traceback
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ccxt

from optimal_trading_analyzer import OptimalTradingAnalyzer

# ---------------------------
# Config
# ---------------------------
EXCHANGE = "binance"
SYMBOL = "DOGE/USDT"  # يُوستد در بایننس
TIMEFRAME = "1m"
LOOKBACK = 60

# Position sizing
MAX_POSITION_PCT = 1.0   # 100% از موجودی هر مدل
TAKER_FEE = 0.0026       # 0.26% (ورود + خروج اعمال میشه)

# Initial balances for two models
INIT_BALANCE_MODEL1 = 1000.0
INIT_BALANCE_MODEL2 = 1000.0

# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Paths
CALIB_PATH = "models/calibration.json"
META1_PATH = "models/meta_fold1.json"
WEIGHT1_PATH = "models/model_fold1.pt"
META2_PATH = "models/meta_fold2.json"
WEIGHT2_PATH = "models/model_fold2.pt"

STATE_PATH = "live_state.json"  # برای ماندگاری وضعیت پوزیشن/موجودی

# ---------------------------
# Model (must mirror train_nn.EntryProfitNet)
# ---------------------------
class EntryProfitNet(nn.Module):
    def __init__(self, n_features: int, hidden=128, layers=2, dropout=0.2, max_profit_cap=0.30):
        super().__init__()
        self.max_profit_cap = max_profit_cap
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers>1 else 0)
        self.head_cls = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):  # x: (B,L,F)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        logit = self.head_cls(h)
        profit01 = self.head_reg(h)
        profit = profit01 * self.max_profit_cap
        return logit, profit

# ---------------------------
# Utils
# ---------------------------
def load_meta(meta_path):
    meta = json.load(open(meta_path))
    feature_cols = meta["feature_cols"]
    center = np.array(meta["robust_center"]) if meta["robust_center"] is not None else None
    scale = np.array(meta["robust_scale"])
    lookback = meta["lookback"]
    cap = meta.get("max_profit_cap", 0.30)
    return feature_cols, center, scale, lookback, cap

def transform_with_robust_scaler(X, center, scale):
    Xn = X.copy().astype(float)
    if center is not None:
        Xn = Xn - center
    Xn = Xn / scale
    # median impute
    med = np.nanmedian(Xn, axis=0)
    inds = np.where(np.isnan(Xn))
    if len(inds[0]) > 0:
        Xn[inds] = np.take(med, inds[1])
    return Xn

def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print(f"[TG][DRY] {msg}")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print("[TG] Error:", e)

def timeframe_ms(tf="1m"):
    return {"1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000,"1h":3_600_000}.get(tf,60_000)

def now_utc():
    return datetime.now(timezone.utc)

# ---------------------------
# State (per-model)
# ---------------------------
def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {
        "model1": {"balance": INIT_BALANCE_MODEL1, "position": None},
        "model2": {"balance": INIT_BALANCE_MODEL2, "position": None},
        "last_ts": None
    }

def save_state(st):
    json.dump(st, open(STATE_PATH, "w"), indent=2, default=str)

# Position format:
# {
#   "entry_time": ISO,
#   "entry_price": float,
#   "qty": float,
#   "tp_price": float,
#   "sl_price": float
# }

# ---------------------------
# Exchange
# ---------------------------
def load_exchange(name):
    ex = getattr(ccxt, name)({
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"adjustForTimeDifference": True} if name=="binance" else {}
    })
    ex.load_markets()
    if SYMBOL not in ex.markets:
        raise ValueError(f"Symbol {SYMBOL} not in {name}")
    return ex

def fetch_last_candles(ex, symbol, timeframe, n=LOOKBACK):
    """Fetch last n candles (including the most recently closed)."""
    # Fetch extra candles since technical indicators will drop initial rows due to NaN
    # We need at least n valid rows after indicator calculation
    fetch_limit = max(n + 50, 150)  # Fetch enough to ensure we have n valid after indicators
    candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
    df = pd.DataFrame(candles, columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

# ---------------------------
# Feature builder (same as training)
# ---------------------------
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Uses OptimalTradingAnalyzer.calculate_technical_indicators to ensure parity.
    Also adds additional features that were created during training.
    """
    import warnings
    import sys
    from io import StringIO
    
    ana = OptimalTradingAnalyzer(initial_balance=1000)
    # df_raw index must be DatetimeIndex UTC, columns: open high low close volume
    df = df_raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_raw index must be DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # Suppress verbose output from calculate_technical_indicators
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        df = ana.calculate_technical_indicators(df)
    finally:
        sys.stdout = old_stdout
    
    print(f"📊 Calculated technical indicators: {len(df)} valid rows from {len(df_raw)} raw candles")
    
    # Add features that were created during training in create_training_labels
    # These are needed for the model but not part of calculate_technical_indicators
    df['trend_strength'] = df['close'].rolling(10).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
    )
    
    # Use ATR for volatility percentile (ATR is already calculated)
    if 'atr_14' in df.columns:
        df['volatility_percentile'] = df['atr_14'].rolling(100).rank(pct=True)
    else:
        # Fallback: calculate simple volatility
        df['volatility_percentile'] = df['close'].rolling(100).std().rolling(100).rank(pct=True)
    
    df['volume_percentile'] = df['volume'].rolling(100).rank(pct=True)
    
    return df

def make_sequence_matrix(df_feat: pd.DataFrame, feature_cols: list, L: int):
    """Return (L, F) numpy ready for model."""
    # Work with a copy to avoid modifying the original dataframe
    df_work = df_feat.copy()
    
    # Create missing features filled with NaN (will be imputed by scaler transform)
    missing = [c for c in feature_cols if c not in df_work.columns]
    if missing:
        # Only warn on first iteration or if count changes
        for c in missing:
            df_work[c] = np.nan
    
    # Select only the required features in the correct order
    X = df_work[feature_cols].tail(L).values  # (L,F)
    return X

# ---------------------------
# Trading logic on candle close
# ---------------------------
def maybe_close_position(model_name, state, candle):
    """
    Close position if TP or SL hit within this candle.
    Decision order: if both hit within same candle: prioritize TP (optimistic).
    candle row has: open high low close
    """
    pos = state[model_name]["position"]
    if not pos:
        return

    entry_price = pos["entry_price"]
    qty = pos["qty"]
    tp = pos["tp_price"]
    sl = pos["sl_price"]

    high = candle["high"]
    low = candle["low"]
    close_price = candle["close"]
    
    # Show current position status
    current_pnl_pct = ((close_price - entry_price) / entry_price) * 100
    print(f"📍 {model_name}: Position open | Entry={entry_price:.6f} | Current={close_price:.6f} | PnL={current_pnl_pct:+.2f}% | TP={tp:.6f} | SL={sl:.6f}")

    hit_tp = high >= tp
    hit_sl = low <= sl

    exit_price = None
    reason = None
    if hit_tp and hit_sl:
        # priority TP first (you can switch policy)
        exit_price = tp
        reason = "TP&SL same-candle (TP priority)"
    elif hit_tp:
        exit_price = tp
        reason = "TP"
    elif hit_sl:
        exit_price = sl
        reason = "SL"

    if exit_price is not None:
        # fees: entry + exit
        fee_buy = (entry_price * qty) * TAKER_FEE
        fee_sell = (exit_price * qty) * TAKER_FEE
        gross = (exit_price - entry_price) * qty
        net = gross - fee_buy - fee_sell

        state[model_name]["balance"] += net
        state[model_name]["position"] = None

        print(f"🚪 {model_name}: Position CLOSED | Reason: {reason} | Exit={exit_price:.6f} | PnL={net:+.2f} USDT | New Balance=${state[model_name]['balance']:.2f}")
        
        msg = (
            f"✅ <b>Close</b> | <i>{model_name}</i>\n"
            f"Reason: {reason}\n"
            f"Exit @ {exit_price:.6f}\n"
            f"PNL: {net:+.2f} USDT\n"
            f"New Balance: {state[model_name]['balance']:.2f} USDT"
        )
        send_telegram(msg)

def maybe_open_position(model_name, state, prob_entry, pred_profit, last_close):
    """
    If no open position and prob >= tau: open all-in position with TP/SL from calibration.
    """
    calib = state["_calib"][model_name]
    tau = calib["tau"]
    gamma = calib["gamma"]
    
    # Always print decision info
    has_position = state[model_name]["position"] is not None
    status_emoji = "🔒" if has_position else "👁️"
    
    if has_position:
        pos = state[model_name]["position"]
        print(f"{status_emoji} {model_name}: Already in position @ {pos['entry_price']:.6f} | Balance: ${state[model_name]['balance']:.2f}")
        return
    
    # No position, check entry criteria
    signal_status = "🟢 ENTRY!" if prob_entry >= tau else "⚪ No signal"
    print(f"{status_emoji} {model_name}: p={prob_entry:.4f} (tau={tau:.4f}) | predTP={pred_profit*100:.2f}% | {signal_status} | Balance: ${state[model_name]['balance']:.2f}")

    if prob_entry >= tau:
        tp_pct = float(pred_profit)  # e.g., 0.05
        sl_pct = float(gamma) * tp_pct

        entry_price = last_close
        balance = state[model_name]["balance"]
        trade_amt = balance * MAX_POSITION_PCT
        qty = (trade_amt * (1 - TAKER_FEE)) / entry_price  # fee on buy

        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        state[model_name]["position"] = {
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": entry_price,
            "qty": qty,
            "tp_price": tp_price,
            "sl_price": sl_price
        }

        print(f"🚀 {model_name}: Position OPENED | Entry={entry_price:.6f} | TP={tp_price:.6f} (+{tp_pct*100:.2f}%) | SL={sl_price:.6f} (-{sl_pct*100:.2f}%) | Qty={qty:.2f}")
        
        msg = (
            f"🟢 <b>Open</b> | <i>{model_name}</i>\n"
            f"Entry @ {entry_price:.6f}\n"
            f"TP: {tp_price:.6f} (+{tp_pct*100:.2f}%)\n"
            f"SL: {sl_price:.6f} (-{sl_pct*100:.2f}%)\n"
            f"Qty: {qty:.2f}\n"
            f"Balance: {balance:.2f} USDT"
        )
        send_telegram(msg)

# ---------------------------
# Main loop
# ---------------------------
def main():
    # Load meta & calibration
    feature_cols1, center1, scale1, look1, cap1 = load_meta(META1_PATH)
    feature_cols2, center2, scale2, look2, cap2 = load_meta(META2_PATH)

    # Load calibration
    calibs = json.load(open(CALIB_PATH))
    # Map folds to models
    # fold1 → model1, fold2 → model2
    calib1 = next(x for x in calibs if x["fold"] == 1)
    calib2 = next(x for x in calibs if x["fold"] == 2)

    # Load models
    model1 = EntryProfitNet(n_features=len(feature_cols1), max_profit_cap=cap1)
    model1.load_state_dict(torch.load(WEIGHT1_PATH, map_location="cpu"))
    model1.eval()

    model2 = EntryProfitNet(n_features=len(feature_cols2), max_profit_cap=cap2)
    model2.load_state_dict(torch.load(WEIGHT2_PATH, map_location="cpu"))
    model2.eval()

    # Exchange
    ex = load_exchange(EXCHANGE)

    # State
    state = load_state()
    state["_calib"] = {
        "model1": {"tau": calib1["tau"], "gamma": calib1["gamma"]},
        "model2": {"tau": calib2["tau"], "gamma": calib2["gamma"]},
    }

    send_telegram("🤖 Live runner started.")

    # Graceful exit
    running = True
    def handle_sig(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    tf_ms = timeframe_ms(TIMEFRAME)
    
    iteration = 0  # Counter for tracking progress

    while running:
        iteration += 1
        loop_start = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"📊 Iteration #{iteration} | {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'='*60}")
            # 1) Fetch candles (more than LOOKBACK to ensure enough after indicator calculation)
            df = fetch_last_candles(ex, SYMBOL, TIMEFRAME, n=LOOKBACK)
            
            # 2) Build features & sequences (must match training)
            feats = build_features(df)  # adds technicals
            
            # After indicator calculation, some initial rows may be dropped due to NaN
            # Check if we have enough data
            if len(feats) < LOOKBACK:
                print(f"⚠️ Not enough valid data after indicators: {len(feats)}/{LOOKBACK}, skipping cycle...")
                continue
            
            # Get the last candle info (from feats which has cleaned data)
            last_candle_ts = feats.index[-1]
            last_row = feats.iloc[-1].to_dict()
            last_close = float(feats["close"].iloc[-1])

            # 3) Close logic first (on the just-closed candle)
            print(f"🔍 Checking exit signals...")
            maybe_close_position("model1", state, last_row)
            maybe_close_position("model2", state, last_row)

            # 4) Model predictions
            print(f"🔮 Running model predictions...")
            
            # Model 1
            X1 = make_sequence_matrix(feats, feature_cols1, LOOKBACK)
            X1n = transform_with_robust_scaler(X1, center1, scale1)
            x1t = torch.from_numpy(X1n).float().unsqueeze(0)  # (1,L,F)
            with torch.no_grad():
                logit1, prof1 = model1(x1t)
                prob1 = float(torch.sigmoid(logit1).cpu().numpy().reshape(-1)[0])
                pred_profit1 = float(prof1.cpu().numpy().reshape(-1)[0])

            # Model 2
            X2 = make_sequence_matrix(feats, feature_cols2, LOOKBACK)
            X2n = transform_with_robust_scaler(X2, center2, scale2)
            x2t = torch.from_numpy(X2n).float().unsqueeze(0)
            with torch.no_grad():
                logit2, prof2 = model2(x2t)
                prob2 = float(torch.sigmoid(logit2).cpu().numpy().reshape(-1)[0])
                pred_profit2 = float(prof2.cpu().numpy().reshape(-1)[0])
            
            print(f"✅ Predictions complete!")

            # 5) Open logic (if no position)
            print(f"\n💼 Checking entry signals:")
            maybe_open_position("model1", state, prob1, pred_profit1, last_close)
            maybe_open_position("model2", state, prob2, pred_profit2, last_close)

            # 6) Summary
            print(f"\n📊 Summary:")
            print(f"   Price: {last_close:.6f} @ {last_candle_ts}")
            print(f"   Model1 Balance: ${state['model1']['balance']:.2f} | Position: {'YES' if state['model1']['position'] else 'NO'}")
            print(f"   Model2 Balance: ${state['model2']['balance']:.2f} | Position: {'YES' if state['model2']['position'] else 'NO'}")

            # 7) Periodic status message (every N minutes)
            if int(time.time()) % (15*60) < 3:  # حدوداً هر ۱۵ دقیقه
                msg = (
                    f"⏱ <b>Status</b>\n"
                    f"Model1 | p={prob1:.3f} | predTP={pred_profit1*100:.2f}% | Bal={state['model1']['balance']:.2f}\n"
                    f"Model2 | p={prob2:.3f} | predTP={pred_profit2*100:.2f}% | Bal={state['model2']['balance']:.2f}\n"
                    f"Last close: {last_close:.6f} @ {last_candle_ts}"
                )
                send_telegram(msg)

            # 8) Save state
            save_state(state)
            
            # Calculate next run time
            elapsed = time.time() - loop_start
            sleep_s = max(1.0, 60.0 - (elapsed % 60.0))
            next_run = now_utc().replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
            print(f"\n⏰ Iteration #{iteration} complete in {elapsed:.2f}s. Sleeping {sleep_s:.1f}s until {next_run.strftime('%H:%M:%S UTC')}...")

        except Exception as e:
            print(f"\n❌ Error in iteration #{iteration}: {e}")
            traceback.print_exc()
            send_telegram(f"❌ Error: {e}\n<code>{traceback.format_exc()}</code>")
            elapsed = time.time() - loop_start
            sleep_s = max(1.0, 60.0 - (elapsed % 60.0))

        # Sleep until next minute boundary
        time.sleep(sleep_s)

    save_state(state)
    send_telegram("👋 Live runner stopped.")

if __name__ == "__main__":
    main()
