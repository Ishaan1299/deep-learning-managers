import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("Starting data preprocessing...")

# Load datasets
internal_df = pd.read_excel('data/Internal_Bank_Dataset.xlsx')
external_df = pd.read_excel('data/External_Cibil_Dataset.xlsx')

# 1. Handle basic cleaning in both dataframes (handling -99999 missing values)
def handle_missing(df):
    # Replace -99999 with NaN
    df = df.replace(-99999, np.nan)
    return df

internal_df = handle_missing(internal_df)
external_df = handle_missing(external_df)

# Drop rows where target is missing
external_df = external_df.dropna(subset=['Approved_Flag'])

# Merge
df = pd.merge(internal_df, external_df, on='PROSPECTID', how='inner')

# Drop columns with too many NaNs
threshold = 0.3 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Fill remaining numerical NaNs with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical NaNs with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Merged Shape after cleaning: {df.shape}")

# 2. Encode Categorical variables
# Target Variable 'Approved_Flag': P1, P2, P3, P4 -> 0, 1, 2, 3
target_encoder = LabelEncoder()
df['Approved_Flag'] = target_encoder.fit_transform(df['Approved_Flag'])

# One-hot encode remaining categorical variables
df = pd.get_dummies(df, drop_first=True)

# 3. Train Test Split
X = df.drop(['Approved_Flag', 'PROSPECTID'], axis=1)
y = df['Approved_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to pandas for easier saving
X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
X_train_final.to_csv('data/processed/X_train.csv', index=False)
X_test_final.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Preprocessing complete. Data saved to data/processed/")
print(f"Final Train Shape: {X_train_final.shape}")
print(f"Final Test Shape: {X_test_final.shape}")
