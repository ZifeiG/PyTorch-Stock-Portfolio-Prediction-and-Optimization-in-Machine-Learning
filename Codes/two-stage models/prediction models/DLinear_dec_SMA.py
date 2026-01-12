import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# ============ Moving Average Decomposition ============
def moving_average(series, kernel_size):
    """Compute moving average for trend extraction."""
    kernel = np.ones(kernel_size) / kernel_size
    if isinstance(series, torch.Tensor):
        series_np = series.numpy()
    else:
        series_np = series
    trend = np.array([
        np.convolve(s, kernel, mode='same') for s in series_np.T
    ]).T
    return trend

# ============ DLinear Model with Decomposition ============
class DLinearDecomposed(nn.Module):
    def __init__(self, input_dim):
        super(DLinearDecomposed, self).__init__()
        self.linear_trend = nn.Linear(input_dim, input_dim)
        self.linear_seasonal = nn.Linear(input_dim, input_dim)

    def forward(self, x, trend):
        seasonal = x - trend
        out_trend = self.linear_trend(trend)
        out_seasonal = self.linear_seasonal(seasonal)
        return out_trend + out_seasonal

# ============ Parameters ============
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')
data = pd.read_csv(data_path).values # shape: (T, num_stocks)

window_size = 60
stride = 20
kernel_size = 8
num_stocks = data.shape[1]
num_batches = 10 # ///num_batches = (len(data) - window_size - 1) // stride
learning_rate = 0.001
epochs = 50

# ============ Storage ============
mae_all, mse_all = [], []

# ============ Batch Training Loop ============
for i in range(num_batches):
    start = i * stride
    if start + window_size >= len(data):
        print(f"⚠️ Batch {i+1} skipped: index out of range")
        continue

    X_window = data[start:start + window_size]  # shape: (60, 30)
    y_true = data[start + window_size]          # shape: (30,)

    X_train = X_window[:-1]                     # shape: (59, 30)
    y_train = X_window[1:]                      # shape: (59, 30)

    trend_train = moving_average(X_train, kernel_size=kernel_size)

    # Convert to tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    trend_tensor = torch.tensor(trend_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, trend_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16)

    # Model
    model = DLinearDecomposed(input_dim=num_stocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Train
    for epoch in range(epochs):
        for xb, trendb, yb in loader:
            pred = model(xb, trendb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Predict next day (Day 61)
    x_last = X_window[-1]
    trend_last = moving_average(x_last[np.newaxis, :], kernel_size=kernel_size)[0]

    x_tensor = torch.tensor(x_last, dtype=torch.float32).unsqueeze(0)
    trend_tensor = torch.tensor(trend_last, dtype=torch.float32).unsqueeze(0)

    y_pred = model(x_tensor, trend_tensor).detach().numpy().squeeze()

    # Evaluation
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    mae_all.append(mae)
    mse_all.append(mse)

    print(f"[Batch {i+1}] MAE: {mae:.6f}, MSE: {mse:.6f}")

# ============ Final Results ============
print("\n===  Decomposed DLinear(SMA) Results ===")
print("Average MAE:", np.mean(mae_all))
print("Average MSE:", np.mean(mse_all))
