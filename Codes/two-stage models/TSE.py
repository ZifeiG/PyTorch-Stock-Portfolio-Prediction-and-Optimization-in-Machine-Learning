import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List, Dict

# ================================================================
# Reproducibility & device
# ================================================================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# Simple MAE / MSE (avoid sklearn)
# ================================================================
def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

# ================================================================
# Paths
# ================================================================
base_dir = r"C:\Users\19043\Desktop\Thesis"

train_path = os.path.join(base_dir, "Raw Datasets", "train_TSE.csv")
test_path = os.path.join(base_dir, "Raw Datasets", "test_TSE.csv")

save_dir = os.path.join(base_dir, "Codes", "2-Stage")
os.makedirs(save_dir, exist_ok=True)

alpha_results_path = os.path.join(save_dir, "DLinear_EMA_alpha_search_train_TSE.csv")
test_portfolio_results_path = os.path.join(save_dir, "DLinear_Markowitz_test_results_TSE.csv")

assert os.path.exists(train_path), f"Train data not found at: {train_path}"
assert os.path.exists(test_path), f"Test data not found at: {test_path}"

# ================================================================
# EMA Decomposition
# ================================================================
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

# ================================================================
# DLinear Model with Decomposition
# ================================================================
class DLinearDecomposed(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_trend = nn.Linear(input_dim, input_dim)
        self.linear_seasonal = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

# ================================================================
# Covariance estimation for Markowitz
# ================================================================
def estimate_covariance_from_window(X_window: np.ndarray, eps_ridge: float = 1e-6) -> np.ndarray:
    """
    X_window: (W, n), assumed gross returns (~1 ±).
    Convert to simple returns R = X - 1 and compute sample covariance.
    """
    R = X_window - 1.0
    Sigma = np.cov(R, rowvar=False, bias=False)
    n = Sigma.shape[0]
    Sigma = Sigma + eps_ridge * np.eye(n)  # ridge to ensure PSD / stability
    return Sigma

# ================================================================
# Simplex projection (for a >= 0, sum(a) = 1)
# ================================================================
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector v onto the probability simplex:
        { a : a_i >= 0, sum_i a_i = 1 }.
    Implementation of the standard sorting-based algorithm.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v

    # sort v in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # find rho
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    if len(rho) == 0:
        # fallback: uniform
        return np.ones_like(v) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(v) / n
    return w / s

# ================================================================
# Markowitz optimizer via projected gradient ascent
# ================================================================
def optimize_portfolio_markowitz(
    mean_vec: np.ndarray,
    Sigma: np.ndarray,
    delta_risk: float = 1e-4,       # kept for compatibility, not used explicitly
    target_return: Optional[float] = None,
    risk_aversion: float = 10.0,    # γ: larger => more risk averse
    pgd_steps: int = 400,
    lr: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Solve penalized mean-variance Markowitz problem without cvxpy:
        max_a   mean_vec^T a - risk_aversion * a^T Σ a
        s.t.    a >= 0, sum(a) = 1.

    We use simple projected gradient ascent on the simplex.
    """
    mean_vec = np.asarray(mean_vec, dtype=float).reshape(-1)
    n = mean_vec.size
    # symmetrize Sigma
    Sigma = 0.5 * (Sigma + Sigma.T)

    # initialize uniform
    a = np.ones(n, dtype=float) / n

    for _ in range(pgd_steps):
        # gradient of f(a) = μ^T a - γ a^T Σ a
        grad = mean_vec - 2.0 * risk_aversion * (Sigma @ a)
        a = a + lr * grad
        # project onto simplex
        a = project_to_simplex(a)

    var_val = float(a @ Sigma @ a)
    return a, var_val

# ================================================================
# Training DLinear on train set for a given alpha
# ================================================================
def build_train_loader(
    train_data: np.ndarray,
    alpha: float,
    batch_size: int,
) -> Tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (x_t, trend_t -> x_{t+1}) pairs from train_data.

    train_data: (T_train, n) gross returns.
    """
    T_train, n = train_data.shape

    # EMA over full train series
    trend_full = exponential_moving_average(train_data, alpha=alpha)  # (T_train, n)

    # For each t = 0..T_train-2:
    # input: x_t, trend_t
    # label: x_{t+1}
    X = train_data[:-1]           # (T_train-1, n)
    trend_X = trend_full[:-1]     # (T_train-1, n)
    y = train_data[1:]            # (T_train-1, n)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    trend_tensor = torch.tensor(trend_X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_tensor, trend_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return loader, X, trend_X, y

def train_model_for_alpha(
    train_data: np.ndarray,
    alpha: float,
    num_stocks: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
) -> Tuple[DLinearDecomposed, float, float]:
    """
    Train DLinearDecomposed on train_data given EMA alpha.
    Returns: (trained_model, train_MAE, train_MSE)
    """
    loader, X_np, trend_np, y_np = build_train_loader(train_data, alpha, batch_size)

    model = DLinearDecomposed(input_dim=num_stocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, tb, yb in loader:
            pred = model(xb, tb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate in-sample train forecast error
    model.eval()
    with torch.no_grad():
        X_tensor_all = torch.tensor(X_np, dtype=torch.float32, device=device)
        trend_tensor_all = torch.tensor(trend_np, dtype=torch.float32, device=device)
        preds = model(X_tensor_all, trend_tensor_all).cpu().numpy()

    train_mae = mean_absolute_error(y_np, preds)
    train_mse = mean_squared_error(y_np, preds)
    return model, train_mae, train_mse

# ================================================================
# Evaluation on test set: rolling Markowitz with fixed trained model
# ================================================================
def evaluate_on_test(
    model: DLinearDecomposed,
    alpha: float,
    test_data: np.ndarray,
    window_size: int,
    delta_risk: float,
    target_return_for_form2: Optional[float] = None,
) -> Tuple[float, float, float, float, List[Dict]]:
    """
    Rolling evaluation on test_data with fixed trained model.

    For each window [s : s+window_size]:
        - use EMA on this window to get trend_last
        - predict next-day gross returns
        - estimate Sigma from this window
        - run Markowitz to get a_hat (pred-based) and a_star (oracle, true mean)
        - compute realized_return, oracle_return, regret
        - accumulate forecast MAE/MSE and portfolio stats
    """
    T_test, n = test_data.shape
    maes, mses = [], []
    portfolio_records: List[Dict] = []

    model.eval()

    for s in range(0, T_test - window_size):
        # Window and label
        X_window = test_data[s : s + window_size]       # (W, n) gross
        y_true_gross = test_data[s + window_size]       # (n,)    gross

        # EMA & features for prediction
        trend_window = exponential_moving_average(X_window, alpha=alpha)  # (W, n)
        x_last = X_window[-1]
        trend_last = trend_window[-1]

        x_tensor = torch.tensor(x_last, dtype=torch.float32, device=device).unsqueeze(0)
        trend_last_tensor = torch.tensor(trend_last, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            y_pred_gross = model(x_tensor, trend_last_tensor).cpu().numpy().squeeze()  # (n,)

        # Covariance on this window
        Sigma_hat = estimate_covariance_from_window(X_window)

        # Convert gross -> simple returns for mean vector
        mu_pred = y_pred_gross - 1.0
        mu_true = y_true_gross - 1.0

        # Markowitz optimization (penalized form)
        a_hat, var_hat = optimize_portfolio_markowitz(
            mu_pred, Sigma_hat, delta_risk=delta_risk
        )
        a_star, var_star = optimize_portfolio_markowitz(
            mu_true, Sigma_hat, delta_risk=delta_risk
        )

        # Portfolio returns & regret (using gross returns)
        realized_return = float(y_true_gross @ a_hat)
        oracle_return = float(y_true_gross @ a_star)
        regret = oracle_return - realized_return

        portfolio_records.append(
            {
                "test_start_index": s,
                "alpha": float(alpha),
                "y_pred": y_pred_gross.tolist(),
                "y_true": y_true_gross.tolist(),
                "a_hat": a_hat.tolist(),
                "a_star": a_star.tolist(),
                "realized_return": realized_return,
                "oracle_return": oracle_return,
                "regret": regret,
                "var_hat": var_hat,
                "var_star": var_star,
            }
        )

        # Forecast metrics on this next-step vector
        maes.append(mean_absolute_error(y_true_gross, y_pred_gross))
        mses.append(mean_squared_error(y_true_gross, y_pred_gross))

    avg_mae = float(np.mean(maes)) if len(maes) > 0 else float("nan")
    avg_mse = float(np.mean(mses)) if len(mses) > 0 else float("nan")
    std_mae = float(np.std(maes)) if len(maes) > 0 else float("nan")
    std_mse = float(np.std(mses)) if len(mses) > 0 else float("nan")

    return avg_mae, avg_mse, std_mae, std_mse, portfolio_records

# ================================================================
# Main: 2-stage training (train) + rolling Markowitz evaluation (test)
# ================================================================
def main():
    # ------------------------------
    # Load data
    # ------------------------------
    train_data = pd.read_csv(train_path).values  # (T_train, n)
    test_data = pd.read_csv(test_path).values    # (T_test, n)

    T_train, num_stocks = train_data.shape
    T_test, num_stocks_test = test_data.shape
    assert num_stocks == num_stocks_test, "Train/Test feature dims mismatch."

    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # ------------------------------
    # Hyperparameters
    # ------------------------------
    window_size = 30             # for test rolling window (Markowitz & EMA)
    batch_size = 32
    learning_rate = 0.001
    epochs = 50

    # EMA alpha grid (tuned on train via in-sample forecast error)
    alphas = np.round(np.linspace(0.05, 0.95, 19), 2)

    # Markowitz parameters
    delta_risk = 1e-4
    target_return_for_form2: Optional[float] = None  # not used in this PG version

    # ------------------------------
    # Alpha search on train
    # ------------------------------
    alpha_records = []
    best_alpha = None
    best_mae = float("inf")
    best_mse = float("inf")

    for a in alphas:
        print(f"\n[Alpha search] Training DLinear with alpha={a:.2f} ...")
        model_a, train_mae, train_mse = train_model_for_alpha(
            train_data=train_data,
            alpha=a,
            num_stocks=num_stocks,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        print(f"alpha={a:.2f} -> Train MAE={train_mae:.6f}, MSE={train_mse:.6f}")

        alpha_records.append(
            {
                "alpha": float(a),
                "train_MAE": train_mae,
                "train_MSE": train_mse,
            }
        )

        # Select best alpha by (MAE, then MSE)
        if (train_mae < best_mae) or (np.isclose(train_mae, best_mae) and train_mse < best_mse):
            best_mae = train_mae
            best_mse = train_mse
            best_alpha = float(a)

    assert best_alpha is not None, "Alpha search failed to pick a best alpha."

    print("\n=== Best alpha on train (by MAE then MSE) ===")
    print(f"best_alpha = {best_alpha:.2f}, MAE={best_mae:.6f}, MSE={best_mse:.6f}")

    # Save alpha search results
    df_alpha = pd.DataFrame(alpha_records).sort_values(by=["train_MAE", "train_MSE"]).reset_index(drop=True)
    df_alpha.to_csv(alpha_results_path, index=False, encoding="utf-8-sig")
    print(f"Saved train alpha search results to: {alpha_results_path}")

    # ------------------------------
    # Retrain DLinear with best alpha on full train set
    # ------------------------------
    print("\n[Final training] Retraining DLinear on full train with best_alpha ...")
    best_model, final_train_mae, final_train_mse = train_model_for_alpha(
        train_data=train_data,
        alpha=best_alpha,
        num_stocks=num_stocks,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    print(f"Final model train MAE={final_train_mae:.6f}, MSE={final_train_mse:.6f}")

    # ------------------------------
    # Evaluate on test with fixed best_model + best_alpha (2-stage)
    # ------------------------------
    print("\n[Evaluation] Rolling Markowitz on test set ...")
    avg_mae_test, avg_mse_test, std_mae_test, std_mse_test, portfolio_records = evaluate_on_test(
        model=best_model,
        alpha=best_alpha,
        test_data=test_data,
        window_size=window_size,
        delta_risk=delta_risk,
        target_return_for_form2=target_return_for_form2,
    )

    print("\n=== Test forecast metrics (one-step ahead) ===")
    print(f"MAE = {avg_mae_test:.6f} ± {std_mae_test:.6f}")
    print(f"MSE = {avg_mse_test:.6f} ± {std_mse_test:.6f}")

    # ------------------------------
    # Save test portfolio results
    # ------------------------------
    df_port = pd.DataFrame(portfolio_records)
    df_port.to_csv(test_portfolio_results_path, index=False, encoding="utf-8-sig")
    print(f"Saved test Markowitz portfolio results to: {test_portfolio_results_path}")


if __name__ == "__main__":
    main()