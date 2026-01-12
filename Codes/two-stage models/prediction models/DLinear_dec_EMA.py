import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# ============ Repro & Device ============
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ EMA Decomposition ============
def exponential_moving_average(series_np: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    series_np: shape (T, F) or (F,) — when (F,), returns (F,) (no time recursion).
    For T>1, returns EMA over time for each feature.
    """
    series_np = np.asarray(series_np)
    if series_np.ndim == 1:
        # Single time point: EMA equals itself by definition
        return series_np.copy()
    T, F = series_np.shape
    ema = np.zeros_like(series_np, dtype=float)
    ema[0] = series_np[0]
    for t in range(1, T):
        ema[t] = alpha * series_np[t] + (1.0 - alpha) * ema[t - 1]
    return ema

# ============ DLinear Model with Decomposition ============
class DLinearDecomposed(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_trend = nn.Linear(input_dim, input_dim)
        self.linear_seasonal = nn.Linear(input_dim, input_dim)

    def forward(self, x, trend):
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

# ============ Paths ============
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')
assert os.path.exists(data_path), f"Data not found at: {data_path}"
# data_path = os.getcwd()
# data = pd.read_csv(data_path +'/NYSE processed.csv').values

# Save tuning results
save_dir = os.path.join(base_dir, 'Prediction Summary')
os.makedirs(save_dir, exist_ok=True)
save_csv = os.path.join(save_dir, 'DLinear_EMA_alpha_search.csv')

# ============ Parameters ============
data = pd.read_csv(data_path).values  # shape: (T, num_stocks)
T, num_stocks = data.shape

window_size = 60
stride = 20
num_batches = 10
batch_size = 16
learning_rate = 0.001
epochs = 50

# Build valid batch starts safely
starts = []
for i in range(num_batches):
    s = i * stride
    if s + window_size < T:
        starts.append(s)

# ============ Add auto-tuning smoothing factor ============
def train_and_eval_for_alpha(alpha: float):
    maes, mses = [], []
    for s in starts:
        X_window = data[s:s + window_size]        # (W, F)
        y_true = data[s + window_size]            # (F,)

        # Train pairs: (x_t -> x_{t+1}), t = 0..W-2
        X_train = X_window[:-1]                   # (W-1, F)
        y_train = X_window[1:]                    # (W-1, F)

        # Trend for training inputs
        trend_train = exponential_moving_average(X_train, alpha=alpha)  # (W-1, F)

        # To tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        trend_tensor = torch.tensor(trend_train, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

        loader = DataLoader(TensorDataset(X_tensor, trend_tensor, y_tensor),
                            batch_size=batch_size, shuffle=True, drop_last=False)

        # Model
        model = DLinearDecomposed(input_dim=num_stocks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Train
        model.train()
        for _ in range(epochs):
            for xb, tb, yb in loader:
                pred = model(xb, tb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Predict next day (Day 31)
        x_last = X_window[-1]                                   # (F,)
        trend_window = exponential_moving_average(X_window, alpha=alpha)  # (W, F)
        trend_last = trend_window[-1]                           # (F,)  <-- FIX

        x_tensor = torch.tensor(x_last, dtype=torch.float32, device=device).unsqueeze(0)
        trend_last_tensor = torch.tensor(trend_last, dtype=torch.float32, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            y_pred = model(x_tensor, trend_last_tensor).cpu().numpy().squeeze()

        # Metrics
        maes.append(mean_absolute_error(y_true, y_pred))
        mses.append(mean_squared_error(y_true, y_pred))

    return float(np.mean(maes)), float(np.mean(mses)), float(np.std(maes)), float(np.std(mses))

# Linear grid
alphas = np.round(np.linspace(0.05, 0.95, 19), 2)
records = []
for a in alphas:
    avg_mae, avg_mse, std_mae, std_mse = train_and_eval_for_alpha(a)
    records.append({
        "alpha": a,
        "avg_MAE": avg_mae,
        "std_MAE": std_mae,
        "avg_MSE": avg_mse,
        "std_MSE": std_mse
    })
    print(f"alpha={a:.2f} -> MAE {avg_mae:.6f} ± {std_mae:.6f} | MSE {avg_mse:.6f} ± {std_mse:.6f}")

df = pd.DataFrame(records).sort_values(by=["avg_MAE", "avg_MSE"]).reset_index(drop=True)
best = df.iloc[0]
print("\n=== Best alpha (by avg_MAE, then avg_MSE) ===")
print(best)

# Save results table
df.to_csv(save_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved search table to: {save_csv}")
