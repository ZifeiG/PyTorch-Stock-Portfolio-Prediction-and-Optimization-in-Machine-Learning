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
# Utilities: covariance & scaling
# ------------------------------------------------------------------
def estimate_covariance_from_window(
    X_window: np.ndarray,
    eps_ridge: float = 1e-6,
    standardize: bool = False,
    shrink_diag: float = 0.10
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
        R = np.clip(R, -5.0, 5.0)
    Sigma = np.cov(R, rowvar=False, bias=False)

    # Diagonal shrinkage: (1-ρ)Σ + ρ·diag(Σ)
    if shrink_diag > 0:
        diag = np.diag(np.diag(Sigma))
        Sigma = (1.0 - shrink_diag) * Sigma + shrink_diag * diag

    # Apply a mild ridge regularization to avoid singularity
    n = Sigma.shape[0]
    Sigma = Sigma + eps_ridge * np.eye(n)

    return Sigma


# ------------------------------------------------------------------
# Markowitz solvers
# ------------------------------------------------------------------
def _solve_markowitz_socp(mu_hat: np.ndarray, Sigma: np.ndarray, delta_risk: float,
                          prefer: str = "CLARABEL") -> np.ndarray:
    """
    Solve SOCP form:
      max  mu_hat^T x - 0.5 * beta * ||x||^2
      s.t. sum(x)=1, x>=0, x<=cap, x^T Σ x <= delta_risk
    """
    n = mu_hat.size
    x = cp.Variable(n)
    cap_val = 0.15
    beta = 1e-2

    cons = [
        cp.sum(x) == 1,
        x >= 0,
        x <= cap_val,
        cp.quad_form(x, Sigma) <= delta_risk
    ]
    obj = cp.Maximize(mu_hat @ x - 0.5 * beta * cp.sum_squares(x))
    prob = cp.Problem(obj, cons)

    if prefer.upper() == "CLARABEL":
        try:
            prob.solve(
                solver=cp.CLARABEL,
                verbose=False,
                max_iter=10000,
                eps_abs=1e-8,
                eps_rel=1e-8
            )
        except Exception:
            pass

    if (x.value is None) or (prob.status not in ("optimal", "optimal_inaccurate")):
        try:
            prob.solve(
                solver=cp.ECOS,
                verbose=False,
                max_iters=2000,
                abstol=1e-8,
                reltol=1e-8,
                feastol=1e-8,
                warm_start=True
            )
        except Exception:
            pass

    if (x.value is None) or (prob.status not in ("optimal", "optimal_inaccurate")):
        try:
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                max_iters=30000,
                eps=5e-6,
                acceleration_lookback=20,
                use_indirect=True
            )
        except Exception:
            pass

    if x.value is None:
        return np.ones(n) / n

    w = np.clip(np.asarray(x.value).ravel(), 0, 1)
    s = float(w.sum())
    return (w / s) if s > 1e-12 else np.ones(n) / n


def _solve_markowitz_qp_osqp(mu_hat: np.ndarray, Sigma: np.ndarray, gamma: float) -> np.ndarray:
    """
    OSQP QP form (no risk constraint):
      max  mu_hat^T x - gamma * x^T Σ x - 0.5 * beta * ||x||^2
      s.t. sum(x)=1, x>=0
    """
    n = mu_hat.size
    x = cp.Variable(n, nonneg=True)
    Q = (Sigma + Sigma.T) / 2.0
    beta = 1e-2

    obj = cp.Maximize(mu_hat @ x - gamma * cp.quad_form(x, Q) -
                      0.5 * beta * cp.sum_squares(x))
    cons = [cp.sum(x) == 1]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(
            solver=cp.OSQP,
            verbose=False,
            max_iter=20000,
            eps_abs=1e-8,
            eps_rel=1e-8,
            polish=True
        )
    except Exception:
        return np.ones(n) / n

    if x.value is None:
        return np.ones(n) / n

    w = np.clip(np.asarray(x.value).ravel(), 0, 1)
    s = float(w.sum())
    return (w / s) if s > 1e-12 else np.ones(n) / n


def solve_markowitz(mu_hat: np.ndarray,
                    Sigma: np.ndarray,
                    delta_risk: float = 2e-3,
                    solver_choice: str = "SOCP_AUTO",
                    gamma_mv: float = 5.0) -> np.ndarray:
    if solver_choice.upper() == "OSQP_QP":
        return _solve_markowitz_qp_osqp(mu_hat, Sigma, gamma=gamma_mv)
    else:
        return _solve_markowitz_socp(mu_hat, Sigma, delta_risk, prefer="CLARABEL")


# ------------------------------------------------------------------
# SPO+ loss
# ------------------------------------------------------------------
def spo_plus_loss(mu_true_torch: torch.Tensor,
                  mu_pred_torch: torch.Tensor,
                  Sigma: np.ndarray,
                  delta_risk: float,
                  solver_choice: str = "SOCP_AUTO",
                  gamma_mv: float = 5.0) -> tuple[torch.Tensor, np.ndarray]:
    mu_true = mu_true_torch.detach().cpu().numpy()
    mu_pred = mu_pred_torch.detach().cpu().numpy()

    x_star_true = solve_markowitz(mu_true, Sigma, delta_risk,
                                  solver_choice=solver_choice, gamma_mv=gamma_mv)
    shifted = 2.0 * mu_pred - mu_true
    x_star_shift = solve_markowitz(shifted, Sigma, delta_risk,
                                   solver_choice=solver_choice, gamma_mv=gamma_mv)

    x_true_t = torch.tensor(x_star_true, dtype=torch.float32, device=device)
    x_shift_t = torch.tensor(x_star_shift, dtype=torch.float32, device=device)

    loss = torch.dot(mu_true_torch - 2.0 * mu_pred_torch, x_shift_t) \
           + torch.dot(2.0 * mu_pred_torch, x_true_t) \
           - torch.dot(mu_true_torch, x_true_t)

    return loss, x_star_true


# ------------------------------------------------------------------
# Training + Evaluation
# ------------------------------------------------------------------
def train_and_eval_spo(loader: DataLoader,
                       data: np.ndarray,
                       window_size: int = 30,
                       epochs: int = 5,
                       alpha: float = 0.2,
                       delta_risk: float = 2e-3,
                       save_dir: str = "Integration Summary",
                       clamp_pred: float = 0.10,
                       lambda_reg: float = 5e-4,
                       solver_choice: str = "SOCP_AUTO",
                       gamma_mv: float = 5.0,
                       max_updates: int | None = None,
                       mode: str = "train",
                       dataset_name: str = "SP500"):   # <<< minimal add

    for sample in loader:
        xb0 = sample[0]
        num_stocks = xb0.shape[-1]
        break

    model = DLinearDecomposed(input_dim=num_stocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    all_records = []
    updates_done = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for xb, tb, yb, sb in loader:
            xb, tb, yb, sb = xb.to(device), tb.to(device), yb.to(device), sb.to(device)
            mu_pred = model(xb, tb)

            if clamp_pred is not None and clamp_pred > 0:
                mu_pred = torch.tanh(mu_pred) * clamp_pred

            batch_loss = 0.0
            valid_cnt = 0

            for i in range(mu_pred.size(0)):
                start_idx = int(sb[i].item())
                future_window = data[start_idx + 1: start_idx + 1 + window_size]
                if future_window.shape[0] < window_size:
                    continue

                Sigma_true = estimate_covariance_from_window(future_window)

                loss_i, x_star_true = spo_plus_loss(
                    yb[i], mu_pred[i], Sigma_true, delta_risk,
                    solver_choice=solver_choice, gamma_mv=gamma_mv
                )

                if lambda_reg > 0:
                    loss_i = loss_i + lambda_reg * torch.sum(mu_pred[i] ** 2)

                batch_loss = batch_loss + loss_i
                valid_cnt += 1

                model.eval()
                with torch.no_grad():
                    y_true_gross = data[start_idx + window_size]
                    y_pred_gross = mu_pred[i].detach().cpu().numpy() + 1.0

                    a_hat = solve_markowitz(
                        mu_pred[i].detach().cpu().numpy(), Sigma_true, delta_risk,
                        solver_choice=solver_choice, gamma_mv=gamma_mv
                    )
                    a_star = solve_markowitz(
                        yb[i].detach().cpu().numpy(), Sigma_true, delta_risk,
                        solver_choice=solver_choice, gamma_mv=gamma_mv
                    )

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

            if valid_cnt > 0:
                batch_loss = batch_loss / valid_cnt
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += float(batch_loss.item())
                updates_done += 1

                if (max_updates is not None) and (updates_done >= max_updates):
                    print(f"[Early stop] Reached max_updates={max_updates}.")
                    break

        print(f"[{mode.upper()}] Epoch {epoch + 1}/{epochs}, "
              f"Loss={epoch_loss / max(len(loader), 1):.6f}")

        if len(all_records) > 0:
            last_rec = all_records[-1]
            a_hat = np.array(last_rec["a_hat"])
            max_w, mean_w = a_hat.max(), a_hat.mean()
            argmax_w = a_hat.argmax()
            print(f"   [Debug] max weight={max_w:.4f} (asset #{argmax_w}), "
                  f"mean={mean_w:.4e}")
            if max_w > 0.9:
                print("Warning: near all-in allocation detected.")

        # Save intermediate results
        os.makedirs(save_dir, exist_ok=True)
        df_tmp = pd.DataFrame(all_records)
        save_path = os.path.join(save_dir, f"Markowitz_SPO+_{mode}_epoch{epoch + 1}_{dataset_name}.csv")
        df_tmp.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Intermediate {mode} results saved to: {save_path}\n")

        # Save checkpoints
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(save_dir, f"{mode}_checkpoint_epoch{epoch + 1}_{dataset_name}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model checkpoint saved: {ckpt_path}\n")

    # Save final results
    final_path = os.path.join(save_dir, f"Markowitz_SPO+_{mode}_final_{dataset_name}.csv")
    df = pd.DataFrame(all_records)
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"Final {mode} results saved to: {final_path}")
    return model, df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

    dataset_name = "SP500"   # <<< set dataset name here (future: DJIA/SP500/TSE)

    train_path = os.path.join(base_dir, "Raw Datasets", f"train_{dataset_name}.csv")
    test_path = os.path.join(base_dir, "Raw Datasets", f"test_{dataset_name}.csv")
    assert os.path.exists(train_path), f"Train data not found at {train_path}"
    assert os.path.exists(test_path), f"Test data not found at {test_path}"

    print(f"Using training dataset: {train_path}")
    print(f"Using testing dataset:  {test_path}")

    # =================== Training data ===================
    data = pd.read_csv(train_path).values  # gross returns

    window_size = 30
    alpha = 0.2
    start_offset = 10

    T, num_stocks = data.shape
    X_last, Y, Trends, Starts = [], [], [], []

    for t in range(start_offset, T - window_size - 1):
        X_window = data[t:t + window_size]
        trend_window = exponential_moving_average(X_window, alpha=alpha)
        X_last.append(X_window[-1])
        Trends.append(trend_window[-1])
        Y.append(data[t + window_size] - 1.0)  # simple returns
        Starts.append(t)

    X_last = np.array(X_last)
    Trends = np.array(Trends)
    Y = np.array(Y)
    Starts = np.array(Starts, dtype=np.int64)

    dataset = TensorDataset(
        torch.tensor(X_last, dtype=torch.float32),
        torch.tensor(Trends, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(Starts, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ==============================================================
    # Training
    # ==============================================================
    root_save_dir = r"C:\Users\19043\Desktop\Thesis\Results and Plots\Bilevel 1"
    save_dir = os.path.join(root_save_dir, dataset_name)  # <<< per-dataset subfolder
    os.makedirs(save_dir, exist_ok=True)

    model, results_train = train_and_eval_spo(
        loader=loader,
        data=data,
        window_size=window_size,
        epochs=20,
        alpha=alpha,
        delta_risk=2e-3,
        save_dir=save_dir,
        clamp_pred=0.10,
        lambda_reg=5e-4,
        solver_choice="SOCP_AUTO",
        gamma_mv=5.0,
        max_updates=None,
        mode="train",
        dataset_name=dataset_name
    )

    # ==============================================================
    # Testing
    # ==============================================================
    print("\n=== Evaluating on test dataset ===")
    test_data = pd.read_csv(test_path).values

    T_test, num_stocks_test = test_data.shape
    X_last_t, Y_t, Trends_t, Starts_t = [], [], [], []

    for t in range(5, T_test - window_size - 1):
        X_window_t = test_data[t:t + window_size]
        trend_window_t = exponential_moving_average(X_window_t, alpha=alpha)
        X_last_t.append(X_window_t[-1])
        Trends_t.append(trend_window_t[-1])
        Y_t.append(test_data[t + window_size] - 1.0)
        Starts_t.append(t)

    X_last_t = np.array(X_last_t)
    Trends_t = np.array(Trends_t)
    Y_t = np.array(Y_t)
    Starts_t = np.array(Starts_t, dtype=np.int64)

    dataset_t = TensorDataset(
        torch.tensor(X_last_t, dtype=torch.float32),
        torch.tensor(Trends_t, dtype=torch.float32),
        torch.tensor(Y_t, dtype=torch.float32),
        torch.tensor(Starts_t, dtype=torch.long)
    )
    loader_t = DataLoader(dataset_t, batch_size=16, shuffle=False)

    model.eval()
    _, results_test = train_and_eval_spo(
        loader=loader_t,
        data=test_data,
        window_size=window_size,
        epochs=1,
        alpha=alpha,
        delta_risk=2e-3,
        save_dir=save_dir,
        clamp_pred=0.10,
        lambda_reg=0.0,
        solver_choice="SOCP_AUTO",
        gamma_mv=5.0,
        max_updates=None,
        mode="test",
        dataset_name=dataset_name
    )

    # ==============================================================
    # Save train/test (final copies)
    # ==============================================================
    os.makedirs(save_dir, exist_ok=True)

    train_csv = os.path.join(save_dir, f"Markowitz_SPO+_train_final_{dataset_name}.csv")
    test_csv = os.path.join(save_dir, f"Markowitz_SPO+_test_final_{dataset_name}.csv")

    results_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    results_test.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print(f"\nTrain results saved to: {train_csv}")
    print(f"Test results saved to:  {test_csv}")
