import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
# import random
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')
data = pd.read_csv(data_path).values # shape: (5651, 36)

#data_path = r'C:\Users\19043\Desktop\Thesis\Raw Datasets\DJIA_processed.csv'
#data = pd.read_csv(data_path).values  # shape: (507, 30)

# Define models
class DLinear(nn.Module):
    def __init__(self, input_dim):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        return self.linear(x)

class DLinearSliding(nn.Module):
    def __init__(self, input_size, output_size):
        super(DLinearSliding, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)

# Parameters
window_size = 60
stride = 20
sliding_steps = 5
num_stocks = data.shape[1]
num_batches = 10

# Results
mae_basic_all, mse_basic_all = [], []
mae_slide_all, mse_slide_all = [], []

for i in range(num_batches):
    start = i * stride
    X_window = data[start:start + window_size]  # days t to t+59
    y_true = data[start + window_size]          # day t+60 (predict target)

    # -------------------
    # Basic DLinear Model
    # -------------------
    X_basic = X_window[:-1]  # t to t+58
    y_basic = X_window[1:]   # t+1 to t+59

    model_basic = DLinear(input_dim=num_stocks)
    optimizer = torch.optim.Adam(model_basic.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(
        torch.tensor(X_basic, dtype=torch.float32),
        torch.tensor(y_basic, dtype=torch.float32)
    ), batch_size=16)

    for epoch in range(20):
        for xb, yb in loader:
            pred = model_basic(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_pred_basic = model_basic(torch.tensor(X_window[-1], dtype=torch.float32).unsqueeze(0)).detach().numpy().squeeze()

    # ----------------------------
    # Sliding Window DLinear Model
    # ----------------------------
    X_slide = [X_window[j:j+sliding_steps].reshape(-1) for j in range(window_size - sliding_steps)]
    y_slide = X_window[sliding_steps:]

    model_slide = DLinearSliding(input_size=sliding_steps*num_stocks, output_size=num_stocks)
    optimizer = torch.optim.Adam(model_slide.parameters(), lr=0.01)

    loader = DataLoader(TensorDataset(
        torch.tensor(X_slide, dtype=torch.float32),
        torch.tensor(y_slide, dtype=torch.float32)
    ), batch_size=16)

    for epoch in range(20):
        for xb, yb in loader:
            pred = model_slide(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    x_last = X_window[-sliding_steps:].reshape(-1)
    y_pred_slide = model_slide(torch.tensor(x_last, dtype=torch.float32).unsqueeze(0)).detach().numpy().squeeze()

    # ----------
    # Evaluation
    # ----------
    mae_basic = mean_absolute_error(y_true, y_pred_basic)
    mse_basic = mean_squared_error(y_true, y_pred_basic)
    mae_slide = mean_absolute_error(y_true, y_pred_slide)
    mse_slide = mean_squared_error(y_true, y_pred_slide)

    mae_basic_all.append(mae_basic)
    mse_basic_all.append(mse_basic)
    mae_slide_all.append(mae_slide)
    mse_slide_all.append(mse_slide)

# Convert to DataFrame
results_df = pd.DataFrame({
    "Batch": list(range(1, 11)),
    "MAE_Basic": mae_basic_all,
    "MSE_Basic": mse_basic_all,
    "MAE_Sliding": mae_slide_all,
    "MSE_Sliding": mse_slide_all
})

print(results_df)
print("\n=== Averaged Errors ===")
print("Basic DLinear:   MAE =", np.mean(mae_basic_all), " MSE =", np.mean(mse_basic_all))
print("Sliding Window:  MAE =", np.mean(mae_slide_all), " MSE =", np.mean(mse_slide_all))

# ------------------------
# Repeat prediction batches (robust version)
# ------------------------
max_possible_batches = (len(data) - window_size - 1) // stride
num_batches = min(22, max_possible_batches)

print(f"Running {num_batches} batches (max possible: {max_possible_batches})")

# Results
mae_basic_all, mse_basic_all = [], []
mae_slide_all, mse_slide_all = [], []

for i in range(num_batches):
    start = i * stride
    X_window = data[start:start + window_size]
    if start + window_size >= len(data):
        print(f"⚠️  Batch {i+1} skipped: index out of range")
        continue

    y_true = data[start + window_size]

    # Basic DLinear
    X_basic = X_window[:-1]
    y_basic = X_window[1:]

    model_basic = DLinear(input_dim=num_stocks)
    optimizer = torch.optim.Adam(model_basic.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(
        torch.tensor(X_basic, dtype=torch.float32),
        torch.tensor(y_basic, dtype=torch.float32)
    ), batch_size=16)

    for epoch in range(20):
        for xb, yb in loader:
            pred = model_basic(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_pred_basic = model_basic(torch.tensor(X_window[-1], dtype=torch.float32).unsqueeze(0)).detach().numpy().squeeze()

    # Sliding Window DLinear
    X_slide = [X_window[j:j+sliding_steps].reshape(-1) for j in range(window_size - sliding_steps)]
    y_slide = X_window[sliding_steps:]

    if len(X_slide) != len(y_slide):
        print(f"⚠️  Batch {i+1} skipped: sliding X/Y mismatch ({len(X_slide)} vs {len(y_slide)})")
        continue

    model_slide = DLinearSliding(input_size=sliding_steps*num_stocks, output_size=num_stocks)
    optimizer = torch.optim.Adam(model_slide.parameters(), lr=0.01)

    loader = DataLoader(TensorDataset(
        torch.tensor(X_slide, dtype=torch.float32),
        torch.tensor(y_slide, dtype=torch.float32)
    ), batch_size=16)

    for epoch in range(20):
        for xb, yb in loader:
            pred = model_slide(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    x_last = X_window[-sliding_steps:].reshape(-1)
    y_pred_slide = model_slide(torch.tensor(x_last, dtype=torch.float32).unsqueeze(0)).detach().numpy().squeeze()

    # Evaluation
    mae_basic = mean_absolute_error(y_true, y_pred_basic)
    mse_basic = mean_squared_error(y_true, y_pred_basic)
    mae_slide = mean_absolute_error(y_true, y_pred_slide)
    mse_slide = mean_squared_error(y_true, y_pred_slide)

    mae_basic_all.append(mae_basic)
    mse_basic_all.append(mse_basic)
    mae_slide_all.append(mae_slide)
    mse_slide_all.append(mse_slide)

    print(f"[Batch {i+1:>2}] MAE_Basic: {mae_basic:.6f}, MSE_Basic: {mse_basic:.6f} | "
          f"MAE_Sliding: {mae_slide:.6f}, MSE_Sliding: {mse_slide:.6f}")

results_df = pd.DataFrame({
    "Batch": list(range(1, len(mae_basic_all)+1)),
    "MAE_Basic": mae_basic_all,
    "MSE_Basic": mse_basic_all,
    "MAE_Sliding": mae_slide_all,
    "MSE_Sliding": mse_slide_all
})

print("\n=== Averaged Errors ===")
print("Basic DLinear:   MAE =", np.mean(mae_basic_all), " MSE =", np.mean(mse_basic_all))
print("Sliding Window:  MAE =", np.mean(mae_slide_all), " MSE =", np.mean(mse_slide_all))