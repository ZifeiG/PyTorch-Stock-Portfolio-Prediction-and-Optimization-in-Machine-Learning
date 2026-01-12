from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd().parents[2]
DATA_PATH = BASE / 'Raw Datasets' / 'DJIA_processed.csv'
assert DATA_PATH.exists(), f'DJIA_processed.csv not found at: {DATA_PATH}'
print(f'Data path: {DATA_PATH}')

data = pd.read_csv(DATA_PATH)
print(f'Loaded DJIA data: {data.shape}')

# 按时间顺序划分
n_train = int(len(data) * 0.8)
train_df = data.iloc[:n_train].copy()
test_df  = data.iloc[n_train:].copy()

print(f'\nTrain: {train_df.shape} | Test: {test_df.shape}')

SAVE_DIR = BASE / 'Raw Datasets'
train_path = SAVE_DIR / 'train_DJIA.csv'
test_path  = SAVE_DIR / 'test_DJIA.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f'\nSaved train -> {train_path}')
print(f'Saved test  -> {test_path}')
