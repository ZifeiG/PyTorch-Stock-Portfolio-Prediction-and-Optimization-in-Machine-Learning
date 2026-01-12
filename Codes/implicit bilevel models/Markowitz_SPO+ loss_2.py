import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# Repro & device
# ------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------
# EMA decomposition
# ------------------------------------------------------------------
def exponential_moving_average(series_np: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Compute exponential moving average (EMA)."""
    if series_np.ndim == 1:
        return series_np.copy()
    T, F = series_np.shape
    ema = np.zeros_like(series_np, dtype=float)
    ema[0] = series_np[0]
    for t in range(1, T):
        ema[t] = alpha * series_np[t] + (1.0 - alpha) * ema[t - 1]
    return ema


# ------------------------------------------------------------------
# Simple DLinear with decomposition
# ------------------------------------------------------------------
class DLinearDecomposed(nn.Module):
    """Decomposed linear model with separate trend and seasonal components."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_trend = nn.Linear(input_dim, input_dim)
        self.linear_seasonal = nn.Linear(input_dim, input_dim)

    def forward(self, x, trend):
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


# ------------------------------------------------------------------
# Sigma estimation on a window (GROUND TRUTH window passed in)
# ------------------------------------------------------------------
def estimate_covariance_from_window(
    X_window: np.ndarray,
    eps_ridge: float = 1e-6,
    standardize: bool = True,
) -> np.ndarray:
    """
    Estimate covariance on simple returns of a given window.
    X_window: (W, n) gross returns (~1±).
    """
    R = X_window - 1.0  # simple returns
    if standardize:
        mu = R.mean(axis=0, keepdims=True)
        std = R.std(axis=0, keepdims=True)
        R = (R - mu) / (std + 1e-8)
        R = np.clip(R, -5.0, 5.0)  # robust clamp
    Sigma = np.cov(R, rowvar=False, bias=False)
    n = Sigma.shape[0]
    Sigma = Sigma + eps_ridge * np.eye(n)  # ridge to ensure PSD
    return Sigma


# ------------------------------------------------------------------
# Markowitz solvers: fast (train) vs accurate (eval)
# ------------------------------------------------------------------
def _solve_markowitz_template(mu_hat: np.ndarray, Sigma: np.ndarray, delta_risk: float,
                              solver: str, solver_kwargs: dict) -> np.ndarray:
    """
    Internal helper to solve:
      max  mu_hat^T x
      s.t. sum(x)=1, x>=0, x^T Σ x <= delta_risk
    """
    n = mu_hat.size
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1, x >= 0, cp.quad_form(x, Sigma) <= delta_risk]
    obj = cp.Maximize(mu_hat @ x)
    prob = cp.Problem(obj, cons)

    try:
        if solver == "ECOS":
            prob.solve(solver=cp.ECOS, **solver_kwargs)
        elif solver == "SCS":
            prob.solve(solver=cp.SCS, **solver_kwargs)
        else:
            prob.solve(**solver_kwargs)
    except Exception:
        return np.ones(n) / n  # fallback: uniform

    if x.value is None:
        return np.ones(n) / n

    w = np.clip(x.value, 0, 1)
    s = float(w.sum())
    return (w / s) if s > 1e-12 else np.ones(n) / n


def solve_markowitz_fast(mu_hat: np.ndarray, Sigma: np.ndarray, delta_risk: float = 1e-2) -> np.ndarray:
    """Fast/rough solver for training (favor speed)."""
    return _solve_markowitz_template(
        mu_hat, Sigma, delta_risk,
        solver="SCS",
        solver_kwargs=dict(verbose=False, max_iters=200, eps=1e-3),
    )


def solve_markowitz_accurate(mu_hat: np.ndarray, Sigma: np.ndarray, delta_risk: float = 1e-2) -> np.ndarray:
    """Accurate solver for evaluation (favor correctness)."""
    w = _solve_markowitz_template(
        mu_hat, Sigma, delta_risk,
        solver="ECOS",
        solver_kwargs=dict(verbose=False, abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=10000),
    )
    if w is None:
        w = _solve_markowitz_template(
            mu_hat, Sigma, delta_risk,
            solver="SCS",
            solver_kwargs=dict(verbose=False, max_iters=10000, eps=1e-6),
        )
    return w


# ------------------------------------------------------------------
# SPO+ loss (surrogate), gradient flows through mu_pred_torch
# ------------------------------------------------------------------
def spo_plus_loss(mu_true_torch: torch.Tensor,
                  mu_pred_torch: torch.Tensor,
                  Sigma: np.ndarray,
                  delta_risk: float) -> tuple[torch.Tensor, np.ndarray]:
    """
    SPO+ surrogate loss for one sample.
    - mu_true_torch, mu_pred_torch: (n,) torch tensors
    - Sigma: numpy PSD matrix estimated from GROUND TRUTH future window
    """
    # numpy views (no grad)
    mu_true = mu_true_torch.detach().cpu().numpy()
    mu_pred = mu_pred_torch.detach().cpu().numpy()

    # x* from solvers (constants in torch graph)
    x_star_true = solve_markowitz_fast(mu_true, Sigma, delta_risk)
    shifted = 2.0 * mu_pred - mu_true
    x_star_shift = solve_markowitz_fast(shifted, Sigma, delta_risk)

    x_true_t = torch.tensor(x_star_true, dtype=torch.float32, device=device)
    x_shift_t = torch.tensor(x_star_shift, dtype=torch.float32, device=device)

    # SPO+ surrogate; keep gradient wrt mu_pred_torch
    loss = torch.dot(mu_true_torch - 2.0 * mu_pred_torch, x_shift_t) \
           + torch.dot(2.0 * mu_pred_torch, x_true_t) \
           - torch.dot(mu_true_torch, x_true_t)

    return loss, x_star_true


# ------------------------------------------------------------------
# Training + Evaluation
# ------------------------------------------------------------------
def train_and_eval_spo(data: np.ndarray,
                       window_size: int = 30,
                       epochs: int = 5,
                       batch_size: int = 4,
                       alpha: float = 0.2,
                       delta_risk: float = 1e-4,
                       save_dir: str = "Integration Summary",
                       clamp_pred: float = 0.05,     # clamp mu_pred to ±5%
                       lambda_reg: float = 0.0       # optional L2 reg on mu_pred
                       ):

    T, num_stocks = data.shape
    X_last, Y, Trends, Starts = [], [], [], []

    # ---------------- Build dataset ----------------
    for t in range(T - window_size - 1):
        X_window = data[t:t + window_size]
        trend_window = exponential_moving_average(X_window, alpha=alpha)

        X_last.append(X_window[-1])                 # last-day gross in window
        Trends.append(trend_window[-1])             # last EMA trend in window
        Y.append(data[t + window_size])             # GROUND TRUTH next-day gross
        Starts.append(t)                            # align index

    X_last = np.array(X_last)
    Trends = np.array(Trends)
    Y = np.array(Y) - 1.0                           # convert to simple returns
    Starts = np.array(Starts, dtype=np.int64)

    dataset = TensorDataset(torch.tensor(X_last, dtype=torch.float32),
                            torch.tensor(Trends, dtype=torch.float32),
                            torch.tensor(Y, dtype=torch.float32),
                            torch.tensor(Starts, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------------- Model & optimizer ----------------
    model = DLinearDecomposed(input_dim=num_stocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    all_records = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for xb, tb, yb, sb in loader:
            xb, tb, yb, sb = xb.to(device), tb.to(device), yb.to(device), sb.to(device)
            mu_pred = model(xb, tb)                 # (B, n)

            # clamp predicted simple returns to avoid explosion (±5%)
            if clamp_pred is not None and clamp_pred > 0:
                mu_pred = torch.tanh(mu_pred) * clamp_pred

            batch_loss = 0.0
            for i in range(mu_pred.size(0)):
                start_idx = int(sb[i].item())

                # ===== GROUND TRUTH covariance: use future window (t+1 : t+window_size+1)
                future_window = data[start_idx + 1: start_idx + 1 + window_size]
                if future_window.shape[0] < window_size:
                    continue  # skip tail
                Sigma_true = estimate_covariance_from_window(future_window)

                # ===== SPO+ training loss =====
                loss_i, x_star_true = spo_plus_loss(yb[i], mu_pred[i], Sigma_true, delta_risk)

                # optional L2 regularization on mu_pred
                if lambda_reg > 0:
                    loss_i = loss_i + lambda_reg * torch.sum(mu_pred[i] ** 2)

                batch_loss = batch_loss + loss_i

                # ===== Evaluation (ACCURATE solver) =====
                model.eval()
                with torch.no_grad():
                    # ground-truth next-day gross
                    y_true_gross = data[start_idx + window_size]

                    # predicted gross for reporting
                    y_pred_gross = mu_pred[i].detach().cpu().numpy() + 1.0

                    a_hat = solve_markowitz_accurate(
                        mu_pred[i].detach().cpu().numpy(), Sigma_true, delta_risk
                    )
                    a_star = solve_markowitz_accurate(
                        (yb[i].detach().cpu().numpy()), Sigma_true, delta_risk
                    )  # oracle on true mu (ACCURATE)

                    realized_return = float(y_true_gross @ a_hat)
                    oracle_return = float(y_true_gross @ a_star)
                    regret = oracle_return - realized_return

                    all_records.append({
                        "epoch": int(epoch),
                        "index": int(start_idx),
                        "MAE": float(mean_absolute_error(y_true_gross, y_pred_gross)),
                        "MSE": float(mean_squared_error(y_true_gross, y_pred_gross)),
                        "realized_return": realized_return,
                        "oracle_return": oracle_return,
                        "regret": regret,
                        "a_hat": a_hat.tolist(),
                        "a_star": a_star.tolist()
                    })
                model.train()

            if mu_pred.size(0) > 0:
                batch_loss = batch_loss / mu_pred.size(0)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += float(batch_loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss={epoch_loss / max(len(loader),1):.6f}")

    # ---------------- Save results ----------------
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(all_records)
    save_csv = os.path.join(save_dir, "Markowitz_SPO+_2.csv")
    df.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"Saved results to: {save_csv}")

    return model, df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
    data_path = os.path.join(base_dir, "Raw Datasets", "NYSE_processed.csv")
    assert os.path.exists(data_path), f"Data not found at {data_path}"
    print(f"Using dataset: {data_path}")

    data = pd.read_csv(data_path).values  # gross returns

    model, results = train_and_eval_spo(
        data,
        window_size=30,
        epochs=5,
        batch_size=4,
        alpha=0.2,
        delta_risk=1e-2,
        save_dir="",
        clamp_pred=0.05,      # Predictive limiting (±5%)
        lambda_reg=0.0        # L2 Regularization Strength
    )