"""
UPI Fraud Detection - Data Preparation Script
Loads NPCI monthly aggregate statistics and generates synthetic
transaction-level data with realistic SEQUENTIAL fraud patterns
suited for LSTM sequence learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Load & Parse NPCI Monthly Statistics
# ─────────────────────────────────────────────
print("Loading NPCI monthly UPI statistics...")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
UPI_DIR  = os.path.join(PROJ_DIR, '..', 'UPI_Fraud')

monthly_records = []
for yr_tag in ['2021-22', '2022-23', '2023-24']:
    fpath = os.path.join(UPI_DIR,
        f'Product-Statistics-UPI-Upi-monthly-statistics-{yr_tag}-monthly.xlsx')
    raw = pd.read_excel(fpath, header=None)
    raw.columns = ['Month', 'Volume_Mn', 'Avg_Daily_Volume_Mn', 'Value_Cr', 'Avg_Daily_Value_Cr']
    raw = raw.iloc[1:].dropna(subset=['Month']).reset_index(drop=True)
    monthly_records.append(raw)

monthly_df = pd.concat(monthly_records, ignore_index=True)
for col in ['Volume_Mn', 'Value_Cr']:
    monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')
monthly_df = monthly_df.dropna()

avg_txn_value_inr = (monthly_df['Value_Cr'].mean() * 1e7) / \
                    (monthly_df['Volume_Mn'].mean() * 1e6)
print(f"Loaded {len(monthly_df)} monthly records.")
print(f"Calibrated avg per-transaction amount: Rs.{avg_txn_value_inr:.2f}\n")

# ─────────────────────────────────────────────
# 2. Generate User Sessions with Sequential Patterns
# ─────────────────────────────────────────────
# Each sample = a SEQUENCE of SEQ_LEN=15 transactions for one user
# Label = 1 (fraud) if the session is a fraud session, else 0
#
# FRAUD session pattern (sequential cues the LSTM can learn):
#   t1-t8 : Normal low-amount transactions, known payees, regular hours
#   t9    : Device change OR SIM swap signal (device_changed=1)
#   t10-12: Velocity spike — many transactions in quick succession
#   t13   : Large amount to a new payee
#   t14   : Very large amount to another new payee (unusual hour)
#   t15   : Largest amount — to new payee at late night  ← FRAUD label
#
# NORMAL session: all 15 steps look typical, no escalation

N_SESSIONS = 80_000    # total user-sessions
FRAUD_RATE  = 0.23     # 23% fraud sessions (more balanced for training)
SEQ_LEN     = 15
FEATURE_COLS = [
    'amount', 'hour_of_day', 'day_of_week', 'is_weekend',
    'is_late_night', 'is_new_payee', 'device_changed',
    'high_amount_flag', 'txn_velocity', 'time_since_last',
    'cumulative_amount_ratio',   # ratio of this txn to user's historical avg
    'payee_familiarity',         # 0=new payee, 1=known payee (inverse of is_new_payee with history)
]
N_FEATURES = len(FEATURE_COLS)

n_fraud  = int(N_SESSIONS * FRAUD_RATE)
n_normal = N_SESSIONS - n_fraud

print(f"Generating {N_SESSIONS:,} user sessions "
      f"({n_fraud:,} fraud, {n_normal:,} normal)...")

mu_log    = np.log(avg_txn_value_inr) - 0.5
sigma_log = 1.3

def normal_txn(t, user_avg_amount, rng):
    """Generate one step of a normal transaction sequence."""
    amount   = rng.lognormal(mean=np.log(user_avg_amount) - 0.1, sigma=0.5)
    amount   = float(np.clip(amount, 10, 30_000))
    hour     = int(rng.integers(7, 22))                    # daytime hours
    dow      = int(rng.integers(0, 7))
    weekend  = int(dow >= 5)
    late     = 0
    new_pay  = int(rng.random() < 0.05)                    # rarely new payee
    dev_chg  = int(rng.random() < 0.01)                    # very rare device change
    high_amt = int(amount > 50_000)
    velocity = int(rng.integers(1, 4))                     # low velocity
    since    = float(rng.exponential(scale=300))           # long gap between txns
    since    = float(np.clip(since, 10, 1440))
    cum_rat  = float(amount / user_avg_amount)             # close to 1.0
    payee_f  = 1.0 - new_pay
    return [amount, hour, dow, weekend, late, new_pay,
            dev_chg, high_amt, velocity, since, cum_rat, payee_f]

def fraud_txn_sequence(user_avg_amount, rng):
    """
    Generate a 15-step sequence with a clear fraud escalation pattern.
    Steps 1-8 : normal  (LSTM baseline)
    Step  9   : device change signal
    Steps 10-12: velocity spike
    Steps 13-15: large amounts to new payees at unusual hours
    """
    seq = []
    # Steps 1-8: normal
    for t in range(8):
        seq.append(normal_txn(t, user_avg_amount, rng))

    # Step 9: device change (account takeover signal)
    a = rng.lognormal(mean=np.log(user_avg_amount), sigma=0.3)
    seq.append([float(np.clip(a, 10, 30_000)),
                int(rng.integers(7, 22)), int(rng.integers(0, 7)), 0, 0,
                0, 1,   # device_changed = 1 ← key signal
                0, int(rng.integers(1, 3)), float(rng.exponential(300)),
                float(a / user_avg_amount), 1.0])

    # Steps 10-12: velocity spike (rapid transactions)
    for t in range(3):
        a = rng.lognormal(mean=np.log(user_avg_amount) + 0.3, sigma=0.4)
        seq.append([float(np.clip(a, 10, 40_000)),
                    int(rng.integers(7, 22)), int(rng.integers(0, 7)), 0, 0,
                    int(t >= 1),   # payees becoming new
                    0, 0,
                    int(rng.integers(5, 8)),   # high velocity ← key signal
                    float(rng.uniform(1, 15)), # very short time since last
                    float(a / user_avg_amount), float(1 - (t >= 1))])

    # Steps 13-14: moderately elevated amounts, new payees
    for multiplier in [3, 5]:
        a = user_avg_amount * multiplier * rng.uniform(0.8, 1.2)
        a = float(np.clip(a, 5_000, 4_00_000))
        hour = int(rng.integers(22, 24)) if rng.random() < 0.6 else int(rng.integers(0, 5))
        seq.append([a, hour, int(rng.integers(0, 7)), 0, 1,   # late_night=1
                    1, 0, int(a > 50_000),                    # new_payee=1, high_amount
                    int(rng.integers(6, 10)),                  # high velocity
                    float(rng.uniform(0.5, 5.0)),             # very short gap
                    float(a / user_avg_amount), 0.0])

    # Step 15: final fraud transaction — elevated, often late night, new payee
    a = user_avg_amount * float(rng.uniform(4, 9))
    a = float(np.clip(a, 5_000, 2_00_000))
    is_ln = int(rng.random() < 0.55)
    hour  = int(rng.choice([1, 2, 3, 23])) if is_ln else int(rng.integers(7, 22))
    seq.append([a, hour, int(rng.integers(0, 7)), 0, is_ln,
                int(rng.random() < 0.80),   # 80% new payee
                0, int(a > 50_000),
                int(rng.integers(5, 8)),
                float(rng.uniform(0.5, 6.0)),
                float(a / user_avg_amount), 0.0])

    return seq   # length = 15

# Build all sequences
all_sequences = []
all_labels    = []

# Normal sessions — with 20% having one realistic "noisy" event
for i in range(n_normal):
    user_avg = float(rng.lognormal(mean=mu_log, sigma=sigma_log))
    user_avg = float(np.clip(user_avg, 100, 20_000))
    seq = [normal_txn(t, user_avg, rng) for t in range(SEQ_LEN)]
    # 20% chance: insert a one-off device change (lost phone)
    if rng.random() < 0.20:
        t_idx = int(rng.integers(0, SEQ_LEN))
        seq[t_idx][6] = 1   # device_changed
    # 15% chance: one large legitimate transaction (salary, rent)
    if rng.random() < 0.15:
        t_idx = int(rng.integers(0, SEQ_LEN))
        seq[t_idx][0] = float(user_avg * rng.uniform(8, 20))
        seq[t_idx][7] = 1   # high_amount_flag
    all_sequences.append(seq)
    all_labels.append(0)

# Fraud sessions — 15% are "subtle" (low-signal) fraud
for i in range(n_fraud):
    user_avg = float(rng.lognormal(mean=mu_log, sigma=sigma_log))
    user_avg = float(np.clip(user_avg, 100, 20_000))
    if rng.random() < 0.15:
        # Subtle fraud: slightly elevated amounts, no device change
        seq = [normal_txn(t, user_avg, rng) for t in range(SEQ_LEN)]
        for t in range(SEQ_LEN - 3, SEQ_LEN):
            seq[t][0] = float(user_avg * rng.uniform(3, 8))  # moderate escalation
            seq[t][5] = 1   # new payee
            seq[t][8] = int(rng.integers(4, 7))              # moderate velocity
    else:
        seq = fraud_txn_sequence(user_avg, rng)
    all_sequences.append(seq)
    all_labels.append(1)

X = np.array(all_sequences, dtype=np.float32)   # (N, SEQ_LEN, N_FEATURES)
y = np.array(all_labels,    dtype=np.int32)

# Add Gaussian noise to continuous features to create realistic overlap
# Feature indices: 0=amount, 8=txn_velocity, 9=time_since_last, 10=cum_ratio
noise_std = np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4, 0.0],
                     dtype=np.float32)
noise = rng.normal(0, 1, size=X.shape).astype(np.float32) * noise_std
X = X + noise
# Keep binary flags clamped to {0,1}
for idx in [3, 4, 5, 6, 7, 11]:
    X[:, :, idx] = np.clip(X[:, :, idx], 0, 1)
# Keep amounts positive
X[:, :, 0] = np.maximum(X[:, :, 0], 1.0)

print(f"Dataset: X={X.shape}  |  Fraud rate: {y.mean()*100:.1f}%")

# ── Label noise: flip 7% of labels to simulate real-world ambiguity ──────
LABEL_NOISE = 0.07
flip_mask = rng.random(len(y)) < LABEL_NOISE
y = y.copy()
y[flip_mask] = 1 - y[flip_mask]
print(f"After {LABEL_NOISE*100:.0f}% label noise: fraud rate = {y.mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. Scale Features
# ─────────────────────────────────────────────
N, T, F = X.shape
X_2d = X.reshape(-1, F)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_2d).reshape(N, T, F).astype(np.float32)

# ─────────────────────────────────────────────
# 4. Train / Test Split & Save
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")

PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

np.save(os.path.join(PROC_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(PROC_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(PROC_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(PROC_DIR, 'y_test.npy'),  y_test)

print("Processed data saved to data/processed/")
print("Data preparation complete.")
