"""
Food Delivery Sentiment Rating Prediction - Data Preparation
Combines Zomato CSV + JSON data, tokenizes cuisine sequences,
and prepares inputs for LSTM-based rating classification.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
FOOD_DIR = os.path.join(PROJ_DIR, '..', 'Food_Delivery')

# ─────────────────────────────────────────────
# 1. Load & Merge CSV + JSON Data
# ─────────────────────────────────────────────
print("Loading Zomato CSV...")
csv_df = pd.read_csv(os.path.join(FOOD_DIR, 'zomato.csv'), encoding='latin1')
# Filter India only (Country Code = 1)
csv_df = csv_df[csv_df['Country Code'] == 1].copy()
csv_df = csv_df.rename(columns={
    'Cuisines'          : 'cuisines',
    'Price range'       : 'price_range',
    'Average Cost for two': 'avg_cost_for_two',
    'Has Online delivery': 'has_online_del',
    'Has Table booking' : 'has_table_book',
    'Rating text'       : 'rating_text',
    'Aggregate rating'  : 'aggregate_rating',
    'Votes'             : 'votes',
    'City'              : 'city',
    'Restaurant Name'   : 'name',
})
csv_df['has_online_del'] = (csv_df['has_online_del'] == 'Yes').astype(int)
csv_df['has_table_book'] = (csv_df['has_table_book'] == 'Yes').astype(int)
csv_df = csv_df[['cuisines','price_range','avg_cost_for_two','has_online_del',
                  'has_table_book','city','rating_text','aggregate_rating','votes','name']]

print(f"CSV India restaurants: {len(csv_df)}")

print("Loading Zomato JSON files...")
json_records = []
for i in range(1, 6):
    fpath = os.path.join(FOOD_DIR, f'file{i}.json')
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        for r in item.get('restaurants', []):
            rest = r.get('restaurant', {})
            loc  = rest.get('location', {})
            ur   = rest.get('user_rating', {})
            json_records.append({
                'name'            : rest.get('name', ''),
                'cuisines'        : rest.get('cuisines', ''),
                'price_range'     : int(rest.get('price_range', 0)),
                'avg_cost_for_two': float(rest.get('average_cost_for_two', 0)),
                'has_online_del'  : int(rest.get('has_online_delivery', 0)),
                'has_table_book'  : int(rest.get('has_table_booking', 0)),
                'city'            : loc.get('city', ''),
                'rating_text'     : ur.get('rating_text', ''),
                'aggregate_rating': float(ur.get('aggregate_rating', 0)),
                'votes'           : int(ur.get('votes', 0)),
            })

json_df = pd.DataFrame(json_records)
print(f"JSON India restaurants: {len(json_df)}")

# Combine
df = pd.concat([csv_df, json_df], ignore_index=True)
print(f"Combined total: {len(df)}")

# ─────────────────────────────────────────────
# 2. Clean & Filter
# ─────────────────────────────────────────────
# Remove "Not rated" — no label to learn
df = df[df['rating_text'].str.strip() != 'Not rated']
df = df[df['rating_text'].notna() & (df['rating_text'].str.strip() != '')]
df = df[df['cuisines'].notna() & (df['cuisines'].str.strip() != '')]
df = df.reset_index(drop=True)

print(f"After removing 'Not rated': {len(df)}")
print("Rating distribution:\n", df['rating_text'].value_counts())

# ─────────────────────────────────────────────
# 3. Tokenize Cuisine Sequences
# ─────────────────────────────────────────────
def parse_cuisines(c_str):
    """Split comma-separated cuisine string into a list of cleaned tokens."""
    tokens = [t.strip().lower() for t in str(c_str).split(',') if t.strip()]
    return tokens

df['cuisine_tokens'] = df['cuisines'].apply(parse_cuisines)

# Build vocabulary from all cuisine tokens
all_tokens = [tok for tokens in df['cuisine_tokens'] for tok in tokens]
freq = Counter(all_tokens)
print(f"\nUnique cuisine types: {len(freq)}")
print("Top 15 cuisines:", [t for t, _ in freq.most_common(15)])

# Assign integer IDs: 0=PAD, 1=UNK, 2..N=actual cuisines
MIN_FREQ = 5   # only include cuisines appearing >= 5 times
vocab = ['<PAD>', '<UNK>'] + [tok for tok, cnt in freq.most_common() if cnt >= MIN_FREQ]
vocab_size = len(vocab)
tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
print(f"Vocabulary size (min_freq={MIN_FREQ}): {vocab_size}")

MAX_SEQ_LEN = 8   # max cuisines per restaurant (pad/truncate)

def encode_sequence(tokens):
    """Encode tokens to integer IDs, pad/truncate to MAX_SEQ_LEN."""
    ids = [tok2idx.get(tok, 1) for tok in tokens]  # 1 = <UNK>
    if len(ids) < MAX_SEQ_LEN:
        ids = ids + [0] * (MAX_SEQ_LEN - len(ids))  # 0 = <PAD>
    else:
        ids = ids[:MAX_SEQ_LEN]
    return ids

df['cuisine_ids'] = df['cuisine_tokens'].apply(encode_sequence)

# ─────────────────────────────────────────────
# 4. Prepare Numerical Features
# ─────────────────────────────────────────────
# Log-transform skewed features
df['votes']           = pd.to_numeric(df['votes'], errors='coerce').fillna(0)
df['avg_cost_for_two']= pd.to_numeric(df['avg_cost_for_two'], errors='coerce').fillna(df['avg_cost_for_two'].median() if hasattr(df['avg_cost_for_two'], 'median') else 500)
df['price_range']     = pd.to_numeric(df['price_range'], errors='coerce').fillna(2)

df['log_votes']       = np.log1p(df['votes'])
df['log_cost']        = np.log1p(df['avg_cost_for_two'])

NUM_FEATURES = ['price_range', 'has_online_del', 'has_table_book', 'log_votes', 'log_cost']

df[NUM_FEATURES] = df[NUM_FEATURES].fillna(0)

scaler = StandardScaler()

# ─────────────────────────────────────────────
# 5. Encode Target Variable
# ─────────────────────────────────────────────
# Map rating_text to integer labels
RATING_MAP = {'Poor': 0, 'Average': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
df['label'] = df['rating_text'].str.strip().map(RATING_MAP)
df = df[df['label'].notna()].reset_index(drop=True)
df['label'] = df['label'].astype(int)

print(f"\nFinal dataset size: {len(df)}")
print("Label distribution:\n", df['label'].value_counts().sort_index())

# ─────────────────────────────────────────────
# 6. Train / Test Split
# ─────────────────────────────────────────────
X_seq = np.array(df['cuisine_ids'].tolist(), dtype=np.int64)      # (N, MAX_SEQ_LEN)
X_num = df[NUM_FEATURES].values.astype(np.float32)
y     = df['label'].values.astype(np.int64)

# Scale numerical features
X_seq_tr, X_seq_te, X_num_tr, X_num_te, y_tr, y_te = train_test_split(
    X_seq, X_num, y, test_size=0.2, stratify=y, random_state=42
)

X_num_tr = scaler.fit_transform(X_num_tr).astype(np.float32)
X_num_te = scaler.transform(X_num_te).astype(np.float32)

print(f"\nTrain: {X_seq_tr.shape[0]:,}  |  Test: {X_seq_te.shape[0]:,}")

# ─────────────────────────────────────────────
# 7. Save Processed Data
# ─────────────────────────────────────────────
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

np.save(os.path.join(PROC_DIR, 'X_seq_train.npy'), X_seq_tr)
np.save(os.path.join(PROC_DIR, 'X_seq_test.npy'),  X_seq_te)
np.save(os.path.join(PROC_DIR, 'X_num_train.npy'), X_num_tr)
np.save(os.path.join(PROC_DIR, 'X_num_test.npy'),  X_num_te)
np.save(os.path.join(PROC_DIR, 'y_train.npy'),     y_tr)
np.save(os.path.join(PROC_DIR, 'y_test.npy'),      y_te)

# Save vocab metadata
import json as _json
meta = {
    'vocab_size'   : vocab_size,
    'vocab'        : vocab[:200],    # save top-200 for reference
    'max_seq_len'  : MAX_SEQ_LEN,
    'num_features' : NUM_FEATURES,
    'num_classes'  : 5,
    'class_names'  : ['Poor', 'Average', 'Good', 'Very Good', 'Excellent'],
}
with open(os.path.join(PROC_DIR, 'metadata.json'), 'w') as f:
    _json.dump(meta, f, indent=2)

print("Processed data saved to data/processed/")
print("Data preparation complete.")
