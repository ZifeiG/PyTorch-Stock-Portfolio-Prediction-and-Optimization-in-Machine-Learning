import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ EMA Decomposition ============
def exponential_moving_average(series_np: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    series_np: shape (T, F) or (F,)
    If 1D (F,), returns itself. If 2D (T, F), returns EMA over time for each feature.
    """
    series_np = np.asarray(series_np)
    if series_np.ndim == 1:
        return series_np.copy()
    T, F = series_np.shape
    ema = np.zeros_like(series_np, dtype=float)
    ema[0] = series_np[0]
    for t in range(1, T):
        ema[t] = alpha * series_np[t] + (1.0 - alpha) * ema[t - 1]
    return ema

# ============ DLinear Decomposed Model ============
class DLinearDecomposed(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_trend = nn.Linear(input_dim, input_dim)
        self.linear_seasonal = nn.Linear(input_dim, input_dim)

    def forward(self, x, trend):
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

# Paths
if '__file__' in globals():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    base_dir = os.path.abspath(os.getcwd())

data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')
assert os.path.exists(data_path), f"Data not found at: {data_path}"

save_dir = os.path.join(base_dir, 'Optimization Summary')
os.makedirs(save_dir, exist_ok=True)
save_csv_alpha = os.path.join(save_dir, 'DLinear_EMA_alpha_search.csv')
save_csv_portfolio = os.path.join(save_dir, 'Portfolio_results_cap+L2.csv')

# load data
data = pd.read_csv(data_path).values  # shape: (T, num_stocks)
T, num_stocks = data.shape

# Parameters
window_size = 30          # use last 30 days -> predict Day 31
stride = 20
num_batches = 10
batch_size = 16
learning_rate = 0.001
epochs = 50

# Smoothing factor tuning grid
alphas = np.round(np.linspace(0.05, 0.95, 19), 2)

# Portfolio controls (tunable)
cap_per_asset = 0.15      # at most 15% per stock
lambda_l2 = 3.0           # L2 reg strength for spreading
# You can later expose these as CLI args or config.

# Build valid batch starts
starts = []
for i in range(num_batches):
    s = i * stride
    # need y_true at s + window_size, so require s + window_size < T
    if s + window_size < T:
        starts.append(s)

# ============ Portfolio Optimization ============
def optimize_portfolio(pred_returns: np.ndarray, cap=0.1, lam=5.0) -> np.ndarray:
    """
    pred_returns: (F,) predicted next-step returns (z_hat)
    max: z_hat^T a - lam ||a||_2^2
    s.t. sum a = 1, 0 <= a <= cap
    """
    pred_returns = np.asarray(pred_returns).reshape(-1)
    n = len(pred_returns)

    a = cp.Variable(n)
    obj = cp.Maximize(pred_returns @ a - lam * cp.sum_squares(a))
    cons = [cp.sum(a) == 1, a >= 0, a <= cap]
    prob = cp.Problem(obj, cons)

    # Try ECOS first; fallback to SCS/OSQP
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if a.value is None:
            raise RuntimeError("ECOS returned None.")
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if a.value is None:
                raise RuntimeError("SCS returned None.")
        except Exception:
            prob.solve(solver=cp.OSQP, verbose=False)

    sol = np.array(a.value).astype(float).reshape(-1)
    # numerical safety: project onto [0,1], then renormalize to sum=1 if any drift
    sol = np.clip(sol, 0.0, 1.0)
    ssum = sol.sum()
    if ssum <= 0:
        # degenerate case: fallback to uniform
        sol = np.ones(n) / n
    else:
        sol = sol / ssum
    return sol

# ============ Training + Evaluation for a given alpha ============
def train_and_eval_for_alpha(alpha: float):
    maes, mses = [], []
    portfolio_records = []

    for s in starts:
        # Window and labels
        X_window = data[s:s + window_size]     # (W, F)
        y_true = data[s + window_size]         # (F,)

        # Train pairs: (x_t -> x_{t+1})
        X_train = X_window[:-1]                # (W-1, F)
        y_train = X_window[1:]                 # (W-1, F)
        trend_train = exponential_moving_average(X_train, alpha=alpha)

        # Tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        trend_tensor = torch.tensor(trend_train, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

        loader = DataLoader(
            TensorDataset(X_tensor, trend_tensor, y_tensor),
            batch_size=batch_size, shuffle=True, drop_last=False
        )

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
        x_last = X_window[-1]                                          # (F,)
        trend_window = exponential_moving_average(X_window, alpha=alpha)
        trend_last = trend_window[-1]                                  # (F,)

        x_tensor = torch.tensor(x_last, dtype=torch.float32, device=device).unsqueeze(0)
        trend_last_tensor = torch.tensor(trend_last, dtype=torch.float32, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            y_pred = model(x_tensor, trend_last_tensor).cpu().numpy().squeeze()  # (F,)

        # ---- Portfolio Optimization (avoid all-in) ----
        a_hat = optimize_portfolio(y_pred, cap=cap_per_asset, lam=lambda_l2)

        # ---- Oracle (FAIR): same LP as predicted case, with ground truth returns ----
        a_star = optimize_portfolio(y_true, cap=cap_per_asset, lam=lambda_l2)

        # ---- Oracle (one-hot) for reference only ----
        # a_star_onehot = np.zeros_like(y_true)
        # a_star_onehot[int(np.argmax(y_true))] = 1.0

        # returns & regrets
        realized_return = float(y_true @ a_hat)
        oracle_return= float(y_true @ a_star)
        # oracle_return_onehot = float(y_true @ a_star_onehot)

        regret = oracle_return - realized_return
        # regret_onehot = oracle_return_onehot - realized_return

        # Collect for later analysis
        portfolio_records.append({
            "start_index": s,
            "alpha": float(alpha),
            "y_pred": y_pred.tolist(),
            "y_true": y_true.tolist(),
            "a_hat": a_hat.tolist(),
            "a_star": a_star.tolist(),
            "realized_return": realized_return,
            "oracle_return": oracle_return,
            "regret": regret,
            "max_weight": float(np.max(a_hat)),
            "l2_concentration": float(np.sum(a_hat**2)),
            "entropy": float(-np.sum(np.where(a_hat > 0, a_hat * np.log(a_hat), 0.0)))
        })

        # Forecast metrics on the vector (per-feature)
        maes.append(mean_absolute_error(y_true, y_pred))
        mses.append(mean_squared_error(y_true, y_pred))

    # Return avg metrics + all portfolio records
    return float(np.mean(maes)), float(np.mean(mses)), float(np.std(maes)), float(np.std(mses)), portfolio_records

# Main grid loop
records = []
all_portfolios = []

for a in alphas:
    avg_mae, avg_mse, std_mae, std_mse, portfolio_records = train_and_eval_for_alpha(a)
    all_portfolios.extend(portfolio_records)
    records.append({
        "alpha": float(a),
        "avg_MAE": avg_mae,
        "std_MAE": std_mae,
        "avg_MSE": avg_mse,
        "std_MSE": std_mse
    })
    print(f"alpha={a:.2f} -> MAE {avg_mae:.6f} ± {std_mae:.6f} | MSE {avg_mse:.6f} ± {std_mse:.6f}")

# Save outputs
df_alpha = pd.DataFrame(records).sort_values(by=["avg_MAE", "avg_MSE"]).reset_index(drop=True)
best = df_alpha.iloc[0]
print("\n=== Best alpha (by avg_MAE, then avg_MSE) ===")
print(best)

df_alpha.to_csv(save_csv_alpha, index=False, encoding="utf-8-sig")
print(f"Saved alpha search to: {save_csv_alpha}")

df_port = pd.DataFrame(all_portfolios)
df_port.to_csv(save_csv_portfolio, index=False, encoding="utf-8-sig")
print(f"Saved portfolio results to: {save_csv_portfolio}")
