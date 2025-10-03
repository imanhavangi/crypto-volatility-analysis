# -*- coding: utf-8 -*-
"""
Train a leak-safe, non-overfitting neural net for:
  (1) Entry classification (is_optimal_entry)
  (2) Profit regression (future_profit_potential) on entry points

Key design:
- Time-based Walk-Forward with purge gap (no leakage)
- Only past/current features (drop target/label columns from X)
- Sequence input with lookback window
- Two-head model (classification + regression)
- Mask regression loss to y_entry==1
- Early stopping, dropout, weight decay, grad clipping
"""

import os, json, math, random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, mean_squared_error

# -------------------------
# Config
# -------------------------
CSV_PATH = "training_data.csv"   # full features (UTC timestamp col first)
TIMESTAMP_COL = "timestamp"

LOOKBACK = 60        # minutes back for each sample window
BATCH_SIZE = 512
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN = 128
LSTM_LAYERS = 2
LAMBDA_PROFIT = 0.5   # weight of regression head in joint loss
GRAD_CLIP = 1.0

FOLDS = 3            # walk-forward splits
PURGE_MIN = LOOKBACK + 30  # purge gap between train and val in minutes

MIN_PROFIT_CAP = 0.0      # clip predicted profit lower bound
MAX_PROFIT_CAP = 0.30     # 30% cap to stabilize training targets; adjust

# Check CUDA availability
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[INFO] CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("[INFO] CUDA not available. Using CPU (install CUDA + torch with GPU support for faster training)")

SEED = 42

# -------------------------
# Repro
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# -------------------------
# Data loading
# -------------------------
df = pd.read_csv(CSV_PATH)
if TIMESTAMP_COL not in df.columns:
    raise ValueError("CSV must include a 'timestamp' column in ISO UTC")

df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

# Identify labels/targets
LABEL_ENTRY = "is_optimal_entry"
LABEL_EXIT  = "is_optimal_exit"  # not used as input or target
TARGET_PROFIT = "future_profit_potential"

if LABEL_ENTRY not in df.columns or TARGET_PROFIT not in df.columns:
    raise ValueError("CSV must have 'is_optimal_entry' and 'future_profit_potential' columns")

# Build feature set: drop timestamp + labels/targets from X
drop_cols = {TIMESTAMP_COL, LABEL_ENTRY, LABEL_EXIT, TARGET_PROFIT}
# Also drop the new columns we added
drop_cols.add("optimal_hold_time")
drop_cols.add("optimal_exit_price")

feature_cols = [c for c in df.columns if c not in drop_cols]

# Optional: drop any suspicious future-derived feats if present by name pattern
future_like = [c for c in feature_cols if c.lower().startswith("future_")]
if future_like:
    # safety: exclude any "future_*" accidental columns
    feature_cols = [c for c in feature_cols if c not in future_like]

print(f"[INFO] Using {len(feature_cols)} features.")

# Convert to numeric matrix
X_raw = df[feature_cols].copy()

# Remove columns that are all NaN
nan_cols = X_raw.columns[X_raw.isna().all()].tolist()
if nan_cols:
    print(f"[INFO] Removing {len(nan_cols)} all-NaN columns: {nan_cols[:5]}...")
    X_raw = X_raw.drop(columns=nan_cols)
    feature_cols = [c for c in feature_cols if c not in nan_cols]

# Remove columns with >90% NaN
high_nan_cols = X_raw.columns[X_raw.isna().mean() > 0.9].tolist()
if high_nan_cols:
    print(f"[INFO] Removing {len(high_nan_cols)} columns with >90% NaN...")
    X_raw = X_raw.drop(columns=high_nan_cols)
    feature_cols = [c for c in feature_cols if c not in high_nan_cols]

print(f"[INFO] Final features: {len(feature_cols)}")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Total samples: {len(df):,}")

y_entry = df[LABEL_ENTRY].astype(int).values
y_profit = df[TARGET_PROFIT].astype(float).clip(MIN_PROFIT_CAP, MAX_PROFIT_CAP).values
ts = df[TIMESTAMP_COL].values

print(f"[INFO] Entry signals: {y_entry.sum():,} ({y_entry.mean()*100:.2f}%)")
print(f"[INFO] Avg profit on entries: {y_profit[y_entry==1].mean():.2%}")

# -------------------------
# Splitter (Walk-Forward with Purge)
# -------------------------
@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray

def build_walk_forward_folds(timestamps: np.ndarray, folds: int) -> List[Fold]:
    n = len(timestamps)
    # Simple equal-chunk split by index (time-ordered)
    # You can switch to date-based thresholds if desired.
    chunk = n // (folds + 1)
    cuts = [chunk*(i+1) for i in range(folds)]  # end indices for folds
    folds_out = []

    start_train = 0
    for i, val_end in enumerate(cuts, 1):
        # validation window: [val_start, val_end)
        val_start = val_end - chunk
        # purge gap before val_start
        purge_gap = PURGE_MIN
        # find index gap by approximately PURGE_MIN minutes based on 1-min bars:
        purge_size = min(purge_gap, val_start - start_train)
        train_end = val_start - purge_size

        if train_end - start_train < LOOKBACK + 1:
            # too small; skip
            continue

        train_idx = np.arange(start_train, train_end)
        val_idx = np.arange(val_start, val_end)
        folds_out.append(Fold(train_idx=train_idx, val_idx=val_idx))

        # Next fold starts training from beginning (expanding window) or rolling?
        # Expanding window usually stabilizes. We'll expand:
        # start_train remains 0
    return folds_out

folds = build_walk_forward_folds(ts, FOLDS)
print(f"[INFO] Built {len(folds)} folds with purge gap={PURGE_MIN} min.")

# -------------------------
# Sequence dataset
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y_e: np.ndarray, y_p: np.ndarray, lookback: int):
        self.X = X
        self.y_e = y_e
        self.y_p = y_p
        self.L = lookback
    def __len__(self):
        return len(self.X) - self.L + 1
    def __getitem__(self, i):
        j = i + self.L - 1
        x_seq = self.X[i:j+1]          # shape (L, F)
        y1 = self.y_e[j]               # entry label for the last step
        y2 = self.y_p[j]               # profit target for the last step
        return torch.from_numpy(x_seq).float(), torch.tensor([y1], dtype=torch.float32), torch.tensor([y2], dtype=torch.float32)

# -------------------------
# Model
# -------------------------
class EntryProfitNet(nn.Module):
    def __init__(self, n_features: int, hidden=HIDDEN, layers=LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head_cls = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)   # logit
        )
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),  # profit %
            nn.Sigmoid()              # in [0,1]; we will scale to MAX_PROFIT_CAP
        )
    def forward(self, x):  # x: (B, L, F)
        out, _ = self.lstm(x)
        h = out[:, -1, :]          # last step
        logit = self.head_cls(h)   # (B,1)
        profit01 = self.head_reg(h)# (B,1) in [0,1]
        profit = profit01 * MAX_PROFIT_CAP
        return logit, profit

# -------------------------
# Training helpers
# -------------------------
def train_one_fold(X_tr, y_e_tr, y_p_tr, X_va, y_e_va, y_p_va, feature_cols, fold_idx: int):
    print(f"\n{'='*70}")
    print(f"Training Fold {fold_idx}")
    print(f"{'='*70}")
    print(f"Train samples: {len(X_tr):,}, Val samples: {len(X_va):,}")
    
    # Fit scaler only on train
    scaler = RobustScaler()
    scaler.fit(X_tr)

    Xtr = scaler.transform(X_tr)
    Xva = scaler.transform(X_va)

    # Fill NaNs if any (after scaling often fewer NaNs; still safeguard)
    tr_median = np.nanmedian(Xtr, axis=0)
    va_median = tr_median.copy()
    inds = np.where(np.isnan(Xtr))
    if len(inds[0]) > 0:
        Xtr[inds] = np.take(tr_median, inds[1])
    inds = np.where(np.isnan(Xva))
    if len(inds[0]) > 0:
        Xva[inds] = np.take(va_median, inds[1])

    # Build sequence datasets; ensure valid indices (discard first LOOKBACK-1)
    ds_tr = SeqDataset(Xtr, y_e_tr, y_p_tr, LOOKBACK)
    ds_va = SeqDataset(Xva, y_e_va, y_p_va, LOOKBACK)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Sequence samples - Train: {len(ds_tr):,}, Val: {len(ds_va):,}")

    model = EntryProfitNet(n_features=Xtr.shape[1]).to(DEVICE)

    # Class imbalance: pos_weight = Nneg/Npos
    pos = y_e_tr[LOOKBACK-1:].sum()
    neg = len(y_e_tr[LOOKBACK-1:]) - pos
    pos_weight_val = float(neg / max(pos, 1))
    print(f"Class balance - Pos: {pos}, Neg: {neg}, pos_weight: {pos_weight_val:.2f}")
    
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=DEVICE))
    huber = nn.SmoothL1Loss(reduction='none')

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float('inf')
    best_state = None
    patience, patience_limit = 0, 7

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for xb, yb_e, yb_p in dl_tr:
            xb = xb.to(DEVICE); yb_e = yb_e.to(DEVICE); yb_p = yb_p.to(DEVICE)
            optim.zero_grad()
            logit, p_pred = model(xb)
            # classification loss
            loss_cls = bce(logit, yb_e)
            # regression masked loss (only where y_entry==1)
            mask = (yb_e > 0.5).float()
            if mask.sum() > 0:
                loss_reg_all = huber(p_pred, yb_p)
                loss_reg = (loss_reg_all * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss_reg = torch.tensor(0.0, device=DEVICE)
            loss = loss_cls + LAMBDA_PROFIT * loss_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(ds_tr)

        # validation
        model.eval()
        va_loss = 0.0
        all_logits, all_p, all_y_e, all_y_p = [], [], [], []
        with torch.no_grad():
            for xb, yb_e, yb_p in dl_va:
                xb = xb.to(DEVICE); yb_e = yb_e.to(DEVICE); yb_p = yb_p.to(DEVICE)
                logit, p_pred = model(xb)
                loss_cls = bce(logit, yb_e)
                mask = (yb_e > 0.5).float()
                if mask.sum() > 0:
                    loss_reg_all = huber(p_pred, yb_p)
                    loss_reg = (loss_reg_all * mask).sum() / (mask.sum() + 1e-6)
                else:
                    loss_reg = torch.tensor(0.0, device=DEVICE)
                loss = loss_cls + LAMBDA_PROFIT * loss_reg
                va_loss += loss.item() * xb.size(0)

                all_logits.append(logit.cpu().numpy())
                all_p.append(p_pred.cpu().numpy())
                all_y_e.append(yb_e.cpu().numpy())
                all_y_p.append(yb_p.cpu().numpy())
        va_loss /= len(ds_va)

        # Early stopping on val loss
        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                "model": model.state_dict(),
                "scaler_center": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
                "scaler_scale": scaler.scale_.tolist(),
                "feature_cols": feature_cols,
            }
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"[Fold {fold_idx}] Early stop at epoch {epoch}")
                break

        # Optional: print quick metrics
        logits = np.concatenate(all_logits).reshape(-1)
        prob_entry = 1/(1+np.exp(-logits))
        y_true_e = np.concatenate(all_y_e).reshape(-1)
        
        # Filter out NaN values before computing metrics
        valid_mask = ~(np.isnan(prob_entry) | np.isnan(y_true_e))
        if valid_mask.sum() > 0:
            prob_entry_clean = prob_entry[valid_mask]
            y_true_e_clean = y_true_e[valid_mask]
            
            ap = average_precision_score(y_true_e_clean, prob_entry_clean) if y_true_e_clean.sum() > 0 else 0.0
            try:
                auc = roc_auc_score(y_true_e_clean, prob_entry_clean)
            except:
                auc = float("nan")
        else:
            ap = 0.0
            auc = float("nan")
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"[Fold {fold_idx}] Ep {epoch:02d} | Tr {tr_loss:.4f} Va {va_loss:.4f} | AUPRC {ap:.3f} AUC {auc:.3f}")

    # save best
    os.makedirs("models", exist_ok=True)
    torch.save(best_state["model"], f"models/model_fold{fold_idx}.pt")
    meta = {
        "feature_cols": feature_cols,
        "robust_center": best_state["scaler_center"],
        "robust_scale": best_state["scaler_scale"],
        "lookback": LOOKBACK,
        "max_profit_cap": MAX_PROFIT_CAP,
    }
    with open(f"models/meta_fold{fold_idx}.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[Fold {fold_idx}] ✅ Saved model and metadata")
    return f"models/model_fold{fold_idx}.pt", f"models/meta_fold{fold_idx}.json"

# -------------------------
# Threshold & SL calibration
# -------------------------
def calibrate_threshold_and_sl(probs: np.ndarray, prof_pred: np.ndarray, y_e: np.ndarray,
                               grid_tau=None, grid_gamma=None):
    """
    probs: predicted prob(entry) on validation
    prof_pred: predicted profit% on validation
    y_e: true entry labels on validation

    We choose:
      - tau: probability threshold to trigger entry
      - gamma: stop-loss multiplier, SL% = gamma * TP_pred%
    Objective: maximize expected return proxy:
      E[ p * TP - (1-p) * SL ] with TP=prof_pred, SL=gamma*prof_pred
    """
    if grid_tau is None: grid_tau = np.linspace(0.5, 0.95, 10)
    if grid_gamma is None: grid_gamma = np.linspace(0.3, 1.2, 10)

    best = {"tau": None, "gamma": None, "score": -1e9}
    for tau in grid_tau:
        mask = probs >= tau
        if mask.sum() < 5:
            continue
        p = probs[mask]
        tp = prof_pred[mask]
        # proxy expected value per trade (no path dependency):
        exp_val = p * tp - (1 - p) * (grid_gamma[:, None] * tp)  # shape (G, N)
        # average across trades -> shape (G,)
        scores = exp_val.mean(axis=1)
        j = np.argmax(scores)
        if scores[j] > best["score"]:
            best = {"tau": float(tau), "gamma": float(grid_gamma[j]), "score": float(scores[j])}
    return best

# -------------------------
# Train across folds + calibrate
# -------------------------
print("\n" + "="*70)
print("Starting Walk-Forward Cross-Validation Training")
print("="*70)

all_calibs = []

for fi, fold in enumerate(folds, 1):
    tr_idx, va_idx = fold.train_idx, fold.val_idx

    # NOTE: Build arrays for this fold
    X_tr = X_raw.iloc[tr_idx].values
    X_va = X_raw.iloc[va_idx].values
    y_e_tr = y_entry[tr_idx]
    y_e_va = y_entry[va_idx]
    y_p_tr = y_profit[tr_idx]
    y_p_va = y_profit[va_idx]

    # Train one fold
    mdl_path, meta_path = train_one_fold(X_tr, y_e_tr, y_p_tr, X_va, y_e_va, y_p_va, feature_cols, fi)

    # Reload model + scaler to predict on validation for calibration
    print(f"\n[Fold {fi}] Running calibration...")
    meta = json.load(open(meta_path))
    center = np.array(meta["robust_center"]) if meta["robust_center"] is not None else None
    scale = np.array(meta["robust_scale"])
    # RobustScaler transform
    def transform(X):
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

    Xva_scaled = transform(X_va)

    # Build val sequences
    def make_sequences(X, y_e, y_p, L):
        xs, ys_e, ys_p = [], [], []
        for i in range(0, len(X)-L+1):
            j = i+L-1
            xs.append(X[i:j+1])
            ys_e.append(y_e[j])
            ys_p.append(y_p[j])
        return np.stack(xs), np.array(ys_e), np.array(ys_p)

    Xva_seq, y_e_seq, y_p_seq = make_sequences(Xva_scaled, y_e_va, y_p_va, LOOKBACK)

    model = EntryProfitNet(n_features=Xva_seq.shape[-1])
    model.load_state_dict(torch.load(mdl_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits, prof = model(torch.from_numpy(Xva_seq).float())
        logits = logits.numpy().reshape(-1)
        prob = 1/(1+np.exp(-logits))
        prof = prof.numpy().reshape(-1)

    # Filter NaN values before calibration
    valid_mask = ~(np.isnan(prob) | np.isnan(prof) | np.isnan(y_e_seq) | np.isnan(y_p_seq))
    prob = prob[valid_mask]
    prof = prof[valid_mask]
    y_e_seq = y_e_seq[valid_mask]
    y_p_seq = y_p_seq[valid_mask]

    # Calibrate tau & gamma on validation of this fold
    calib = calibrate_threshold_and_sl(prob, prof, y_e_seq)
    calib["fold"] = fi

    # Also store some metrics
    ap = float(average_precision_score(y_e_seq, prob)) if y_e_seq.sum() > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_e_seq, prob))
    except:
        auc = float("nan")
    rmse_on_entries = float(math.sqrt(mean_squared_error(y_p_seq[y_e_seq==1], prof[y_e_seq==1]))) if y_e_seq.sum()>0 else float("nan")
    calib["AUPRC"] = ap
    calib["AUC"] = auc
    calib["RMSE_profit_on_entries"] = rmse_on_entries

    print(f"[Fold {fi}] 📊 Calibration Results:")
    print(f"   • Threshold (τ): {calib['tau']:.3f}")
    print(f"   • Stop-Loss multiplier (γ): {calib['gamma']:.2f}")
    print(f"   • AUPRC: {ap:.3f}")
    print(f"   • AUC: {auc:.3f}")
    print(f"   • RMSE on entries: {rmse_on_entries:.4f}")
    all_calibs.append(calib)

# Save fold calibrations
os.makedirs("models", exist_ok=True)
with open("models/calibration.json", "w") as f:
    json.dump(all_calibs, f, indent=2)

print("\n" + "="*70)
print("✅ Training Complete!")
print("="*70)
print("📁 Output files in ./models directory:")
print("   • model_fold{i}.pt - Model weights for each fold")
print("   • meta_fold{i}.json - Scaler params & feature list")
print("   • calibration.json - Threshold & stop-loss calibration")
print("\n💡 Usage:")
print("   1. Load model + meta for inference")
print("   2. Use tau (threshold) to decide entry")
print("   3. Use gamma * predicted_profit for stop-loss")
print("="*70)
