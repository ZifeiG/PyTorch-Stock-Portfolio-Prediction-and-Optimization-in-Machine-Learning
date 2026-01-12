import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ EMA Decomposition ============
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


# ============ DLinear with Decomposition ============
class DLinearDecomposed(nn.Module):
    """Decomposed linear model with separate trend and seasonal components."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_trend = nn.Linear(input_dim, input_dim)
        self.linear_seasonal = nn.Linear(input_dim, input_dim)

    def forward(self, x, trend):
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


# ============ Covariance Estimation ============
def estimate_covariance_from_window(X_window: np.ndarray, eps_ridge: float = 1e-6) -> np.ndarray:
    """Estimate covariance matrix from simple returns of a given window."""
    R = X_window - 1.0
    Sigma = np.cov(R, rowvar=False, bias=False)
    n = Sigma.shape[0]
    Sigma = Sigma + eps_ridge * np.eye(n)  # ridge to ensure PSD
    return Sigma


# ============ Markowitz Solver ============
def solve_markowitz(mu_hat: np.ndarray, Sigma: np.ndarray, delta_risk: float = 1e-4):
    """
    Solve Markowitz portfolio optimization:
    Maximize mu_hat^T x subject to variance and budget constraints.
    Constraints: sum(x)=1, x>=0, x^T Σ x <= delta_risk
    """
    import warnings
    warnings.filterwarnings("ignore")

    n = mu_hat.size
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1, x >= 0, cp.quad_form(x, Sigma) <= delta_risk]
    obj = cp.Maximize(mu_hat @ x)
    prob = cp.Problem(obj, cons)

    try:
        # only SCS, very loose tolerance
        prob.solve(solver=cp.SCS, verbose=False, max_iters=50, eps=1e-2)
    except Exception:
        return np.ones(n) / n  # fallback: uniform

    if x.value is None:
        return np.ones(n) / n

    sol = np.clip(x.value, 0, 1)
    return sol / sol.sum()


# ============ SPO+ Loss ============
def spo_plus_loss(mu_true_torch, mu_pred_torch, Sigma, delta_risk):
    """
    SPO+ surrogate loss for one sample.
    mu_true_torch: torch tensor (n,)
    mu_pred_torch: torch tensor (n,)
    """
    mu_true = mu_true_torch.detach().cpu().numpy()
    mu_pred = mu_pred_torch.detach().cpu().numpy()

    x_star_true = solve_markowitz(mu_true, Sigma, delta_risk)
    shifted = 2 * mu_pred - mu_true
    x_star_shift = solve_markowitz(shifted, Sigma, delta_risk)

    loss_val = torch.dot(mu_true_torch - 2 * mu_pred_torch,
                         torch.tensor(x_star_shift, dtype=torch.float32, device=device)) \
               + 2 * torch.dot(mu_pred_torch,
                               torch.tensor(x_star_true, dtype=torch.float32, device=device)) \
               - torch.dot(mu_true_torch,
                           torch.tensor(x_star_true, dtype=torch.float32, device=device))

    return loss_val, x_star_true

# ============ Training + Evaluation ============
def train_and_eval_spo(data: np.ndarray,
                       window_size: int = 30,
                       epochs: int = 5,
                       batch_size: int = 4,
                       alpha: float = 0.2,
                       delta_risk: float = 1e-4,
                       save_dir: str = "Integration Summary"):

    T, num_stocks = data.shape
    X_last, Y, Trends, Windows = [], [], [], []

    # Build dataset
    for t in range(T - window_size - 1):
        X_window = data[t:t+window_size]
        trend_window = exponential_moving_average(X_window, alpha=alpha)

        X_last.append(X_window[-1])
        Trends.append(trend_window[-1])
        Y.append(data[t+window_size])   # next-day gross return
        Windows.append((X_window, t))

    X_last, Y, Trends = np.array(X_last), np.array(Y), np.array(Trends)
    Y = Y - 1.0  # simple returns

    dataset = TensorDataset(torch.tensor(X_last, dtype=torch.float32),
                            torch.tensor(Trends, dtype=torch.float32),
                            torch.tensor(Y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DLinearDecomposed(input_dim=num_stocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_records = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (xb, tb, yb) in enumerate(loader):
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            mu_pred = model(xb, tb)

            batch_loss = 0.0
            for i in range(mu_pred.size(0)):
                idx = batch_idx * batch_size + i
                if idx >= len(Windows):
                    continue

                _, start_idx = Windows[idx]
                # ===== Ground-truth covariance: use future window (t+1 : t+window_size+1)
                future_window = data[start_idx+1 : start_idx+1+window_size]
                if len(future_window) < window_size:
                    continue
                Sigma_true = estimate_covariance_from_window(future_window)

                # ===== Training loss (SPO+)
                loss_val, x_star_true = spo_plus_loss(yb[i], mu_pred[i], Sigma_true, delta_risk)
                batch_loss = batch_loss + loss_val

                # ===== Evaluation =====
                y_true_gross = data[start_idx+window_size]   # ground-truth gross return
                y_pred_gross = mu_pred[i].detach().cpu().numpy() + 1.0

                a_hat = solve_markowitz(mu_pred[i].detach().cpu().numpy(), Sigma_true, delta_risk)
                a_star = x_star_true

                realized_return = float(y_true_gross @ a_hat)
                oracle_return = float(y_true_gross @ a_star)
                regret = oracle_return - realized_return

                all_records.append({
                    "epoch": epoch,
                    "index": start_idx,
                    "MAE": mean_absolute_error(y_true_gross, y_pred_gross),
                    "MSE": mean_squared_error(y_true_gross, y_pred_gross),
                    "realized_return": realized_return,
                    "oracle_return": oracle_return,
                    "regret": regret,
                    "a_hat": a_hat.tolist(),
                    "a_star": a_star.tolist()
                })

            # === Backpropagation ===
            batch_loss = batch_loss / mu_pred.size(0)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss={epoch_loss/len(loader):.6f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(all_records)
    save_csv = os.path.join(save_dir, "Markowitz_SPO+_1.csv")
    df.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"Saved results to: {save_csv}")

    return model, df


# ============ Main ============
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    data_path = os.path.join(base_dir, "Raw Datasets", "NYSE_processed.csv")
    assert os.path.exists(data_path), f"Data not found at {data_path}"
    print(f"Using dataset: {data_path}")

    data = pd.read_csv(data_path).values  # gross returns

    model, results = train_and_eval_spo(data,
                                        window_size=30,
                                        epochs=5,
                                        batch_size=4,
                                        alpha=0.2)

