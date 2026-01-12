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

# ============ DLinear Model with Decomposition ============
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
save_csv_portfolio = os.path.join(save_dir, 'Portfolio_results_Markowitz.csv')

# Load Data
data = pd.read_csv(data_path).values  # (T, n) gross returns or prices-normalized (≈1±)
T, num_stocks = data.shape

# Parameters
# Forecasting windowing
window_size = 30
stride = 20
num_batches = 10
batch_size = 16
# Training
learning_rate = 0.001
epochs = 50

# EMA tuning grid
alphas = np.round(np.linspace(0.05, 0.95, 19), 2)

# Markowitz parameters
# Form 1: max mean subject to variance budget a^T Σ a <= delta_risk
delta_risk = 1e-4   # tune: 1e-5 ~ 1e-3 typical for daily simple returns
# Form 2 (optional): set target_return=None to use Form 1;
# to use Form 2 (min variance with target mean), set a float later when calling.
target_return_for_form2 = None  # e.g., 0.0005  # average daily return target

# Build valid batch starts
starts = []
for i in range(num_batches):
    s = i * stride
    if s + window_size < T:
        starts.append(s)

# ============ Covariance estimation ============
def estimate_covariance_from_window(X_window: np.ndarray, eps_ridge: float = 1e-6) -> np.ndarray:
    """
    X_window: (W, n), here they look like gross returns (~1±).
    Convert to simple returns R = X - 1 and compute sample covariance.
    """
    R = X_window - 1.0
    Sigma = np.cov(R, rowvar=False, bias=False)
    n = Sigma.shape[0]
    Sigma = Sigma + eps_ridge * np.eye(n)  # ridge to ensure PSD
    return Sigma

# ============ Markowitz optimizer ============
def optimize_portfolio_markowitz(mean_vec: np.ndarray,
                                 Sigma: np.ndarray,
                                 delta_risk: float = 1e-4,
                                 target_return: float | None = None):
    """
    Mean-variance Markowitz:
      - If target_return is None (Form 1): maximize mean subject to variance budget.
      - Else (Form 2): minimize variance subject to mean >= target_return.
    Constraints: sum(a)=1, a>=0 (long-only).
    """
    mean_vec = np.asarray(mean_vec).reshape(-1)
    n = mean_vec.size
    a = cp.Variable(n)
    cons = [cp.sum(a) == 1, a >= 0]

    if target_return is None:
        obj = cp.Maximize(mean_vec @ a)
        cons += [cp.quad_form(a, Sigma) <= delta_risk]
    else:
        obj = cp.Minimize(cp.quad_form(a, Sigma))
        cons += [mean_vec @ a >= target_return]

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if a.value is None:
            raise RuntimeError("ECOS returned None")
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if a.value is None:
                raise RuntimeError("SCS returned None")
        except Exception:
            prob.solve(solver=cp.OSQP, verbose=False)

    a_sol = np.array(a.value, dtype=float).reshape(-1)
    a_sol = np.clip(a_sol, 0.0, 1.0)
    ssum = a_sol.sum()
    if ssum <= 0:
        a_sol = np.ones(n) / n
    else:
        a_sol /= ssum
    var_val = float(a_sol @ Sigma @ a_sol)
    return a_sol, var_val

# ============ Training + Evaluation ============
def train_and_eval_for_alpha(alpha: float):
    maes, mses = [], []
    portfolio_records = []

    for s in starts:
        # Window & labels
        X_window = data[s:s + window_size]     # (W, n) gross
        y_true_gross = data[s + window_size]   # (n,) gross

        # Train pairs: (x_t -> x_{t+1})
        X_train = X_window[:-1]
        y_train = X_window[1:]
        trend_train = exponential_moving_average(X_train, alpha=alpha)

        # Tensors
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

        # Predict next day
        x_last = X_window[-1]
        trend_window = exponential_moving_average(X_window, alpha=alpha)
        trend_last = trend_window[-1]

        x_tensor = torch.tensor(x_last, dtype=torch.float32, device=device).unsqueeze(0)
        trend_last_tensor = torch.tensor(trend_last, dtype=torch.float32, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            y_pred_gross = model(x_tensor, trend_last_tensor).cpu().numpy().squeeze()  # (n,)

        # Estimate Σ on the window
        Sigma_hat = estimate_covariance_from_window(X_window)

        # Convert gross -> simple for Markowitz mean vector
        mu_pred = y_pred_gross - 1.0
        mu_true = y_true_gross - 1.0

        # Predicted portfolio (Markowitz)
        if target_return_for_form2 is None:
            a_hat, var_hat = optimize_portfolio_markowitz(mu_pred, Sigma_hat, delta_risk=delta_risk)
            a_star, var_star = optimize_portfolio_markowitz(mu_true, Sigma_hat, delta_risk=delta_risk)
        else:
            a_hat, var_hat = optimize_portfolio_markowitz(mu_pred, Sigma_hat,
                                                          target_return=float(target_return_for_form2))
            a_star, var_star = optimize_portfolio_markowitz(mu_true, Sigma_hat,
                                                                 target_return=float(target_return_for_form2))

        # Returns & regrets ----
        realized_return = float(y_true_gross @ a_hat)
        oracle_return = float(y_true_gross @ a_star)
        # oracle_return_onehot = float(y_true_gross @ a_star_onehot)

        regret = oracle_return - realized_return
        # regret_onehot = oracle_return_onehot - realized_return

        # Collect results
        portfolio_records.append({
            "start_index": s,
            "alpha": float(alpha),
            "y_pred": y_pred_gross.tolist(),
            "y_true": y_true_gross.tolist(),
            "a_hat": a_hat.tolist(),
            "a_star": a_star.tolist(),
            "realized_return": realized_return,
            "oracle_return": oracle_return,
            "regret": regret,
            "var_hat": var_hat,
            "var_star": var_star
        })

        # Forecast metrics on vectors
        maes.append(mean_absolute_error(y_true_gross, y_pred_gross))
        mses.append(mean_squared_error(y_true_gross, y_pred_gross))

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

# ============ Save outputs ============
df_alpha = pd.DataFrame(records).sort_values(by=["avg_MAE", "avg_MSE"]).reset_index(drop=True)
best = df_alpha.iloc[0]
print("\n=== Best alpha (by avg_MAE, then avg_MSE) ===")
print(best)

df_alpha.to_csv(save_csv_alpha, index=False, encoding="utf-8-sig")
print(f"Saved alpha search to: {save_csv_alpha}")

df_port = pd.DataFrame(all_portfolios)
df_port.to_csv(save_csv_portfolio, index=False, encoding="utf-8-sig")
print(f"Saved portfolio results to: {save_csv_portfolio}")