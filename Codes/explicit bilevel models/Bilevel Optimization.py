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

# -------------------- Simple DLinear -----------------------------
class DLinearSimple(nn.Module):
    """Simple per-asset linear map: R^n -> R^n."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)

# --------------- Lower-level LP (with cap) -----------------------
def solve_unmet_demand_lp(
    d_pred: np.ndarray,
    c: np.ndarray,
    rho: float,
    amax: float,
    cap_factor: float = 1.2
):
    """
    Solve:
        min  c^T a + rho * 1^T u
        s.t. sum(a) <= amax,
             u >= d_pred - a,
             a >= 0, u >= 0,
             a <= cap_factor * d_pred   (optional, controls oversupply)
    """
    n = len(d_pred)
    a = cp.Variable(n, nonneg=True)
    u = cp.Variable(n, nonneg=True)

    cons = [cp.sum(a) <= amax, u >= d_pred - a]
    if cap_factor is not None and cap_factor > 0:
        cons.append(a <= cap_factor * d_pred)

    obj = cp.Minimize(c @ a + rho * cp.sum(u))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-7, reltol=1e-7)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=15000)

    a_val = a.value if a.value is not None else np.zeros(n)
    u_val = np.maximum(d_pred - np.asarray(a_val).ravel(), 0.0)
    return np.asarray(a_val).ravel(), np.asarray(u_val).ravel()

# ---------------------- Training/Evaluation ----------------------
def train_bilevel_demand(
    loader: DataLoader,
    gross_data: np.ndarray,
    cost_vec: np.ndarray,
    amax: float,
    rho: float,
    epochs: int,
    lr: float,
    save_dir: str,
    cap_factor: float,
    demand_floor: float
):
    """Train a demand predictor and evaluate plans under TRUE demand (only key variables)."""
    # Infer input dimension
    for xb, yb, sb in loader:
        input_dim = xb.shape[-1]
        break

    model = DLinearSimple(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.SmoothL1Loss(beta=0.02)

    all_rows = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        regrets = []  # will be appended every sample

        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # Predict demand in [0,1]
            d_pred = torch.sigmoid(model(xb))

            # Supervised prediction loss
            loss = crit(d_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

            # -------- Evaluation under TRUE demand --------
            with torch.no_grad():
                d_pred_np = d_pred.cpu().numpy()
                d_true_np = yb.cpu().numpy()
                sb_np = sb.cpu().numpy()

                for i in range(d_pred_np.shape[0]):
                    # Solve LP for predicted and true demand
                    a_hat, _ = solve_unmet_demand_lp(
                        d_pred_np[i], cost_vec, rho, amax, cap_factor=cap_factor
                    )
                    a_star, _ = solve_unmet_demand_lp(
                        d_true_np[i], cost_vec, rho, amax, cap_factor=cap_factor
                    )

                    # Guard against None/NaN
                    a_hat = np.nan_to_num(np.asarray(a_hat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                    a_star = np.nan_to_num(np.asarray(a_star, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

                    # Evaluate both plans under TRUE demand
                    u_eval_pred = np.maximum(d_true_np[i] - a_hat, 0.0)
                    u_eval_star = np.maximum(d_true_np[i] - a_star, 0.0)

                    denom = max(float(d_true_np[i].sum()), demand_floor)  # > 0 by construction

                    pred_cost = (cost_vec @ a_hat + rho * u_eval_pred.sum()) / denom
                    oracle_cost = (cost_vec @ a_star + rho * u_eval_star.sum()) / denom
                    regret = pred_cost - oracle_cost
                    regrets.append(regret)  # <-- FIX: append so mean is defined

                    # Realized portfolio returns (auxiliary)
                    t = int(sb_np[i])
                    if (t + 1) < gross_data.shape[0]:
                        next_gross = gross_data[t + 1]
                        sum_hat = max(a_hat.sum(), 1e-12)
                        sum_star = max(a_star.sum(), 1e-12)
                        w_hat = a_hat / sum_hat
                        w_star = a_star / sum_star
                        realized_return = float(next_gross @ w_hat)
                        oracle_return   = float(next_gross @ w_star)
                    else:
                        realized_return = np.nan
                        oracle_return   = np.nan

                    # Store only key variables from the note (+ returns for comparison)
                    all_rows.append({
                        "epoch": int(epoch),
                        "index": int(sb_np[i]),
                        "pred_cost": float(pred_cost),
                        "oracle_cost": float(oracle_cost),
                        "regret": float(regret),
                        "a_sum": float(a_hat.sum()),
                        "u_true_sum_predPlan": float(u_eval_pred.sum()),
                        "a_hat": a_hat.tolist(),
                        "a_star": a_star.tolist(),
                        "realized_return": realized_return,
                        "oracle_return": oracle_return,
                    })

        # Epoch summary (safe mean)
        avg_regret = float(np.mean(regrets)) if len(regrets) > 0 else float("nan")
        print(f"[EPOCH {epoch+1}/{epochs}] loss={epoch_loss/len(loader):.6f}, avg_regret={avg_regret:.6f}")

        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(
            os.path.join(save_dir, f"bilevel_epoch{epoch+1}.csv"),
            index=False, encoding="utf-8-sig"
        )

    # Save final results
    df_final = pd.DataFrame(all_rows)
    final_path = os.path.join(save_dir, "bilevel_final.csv")
    df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f" Final results saved to: {final_path}")
    return model, df_final

# ----------------------------- Main ------------------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    data_path = os.path.join(base_dir, "Raw Datasets", "train_NYSE.csv")
    assert os.path.exists(data_path), f"Data not found at {data_path}"
    print(f"Using dataset: {data_path}")

    # Build nonnegative, scaled demand targets
    gross = pd.read_csv(data_path).values
    T, n = gross.shape
    ret = gross[1:] - gross[:-1]
    pos_ret = np.clip(ret, 0.0, None)
    scale = np.percentile(pos_ret, 95) + 1e-8
    demand = np.clip(pos_ret / scale, 0.0, 1.0)
    X = gross[:-1]
    Starts = np.arange(T - 1, dtype=np.int64)

    # Capacity and floor computation
    total_demand = demand.sum(axis=1)
    amax = float(np.median(total_demand) * 0.9)
    amax = max(1e-3, amax)
    demand_floor = float(np.percentile(total_demand, 10))
    demand_floor = max(1.0, demand_floor)

    # Cost parameters
    c = np.ones(n, dtype=float) * 0.05
    rho = 1.0
    cap_factor = 1.2  # allow mild oversupply

    print(f"[INFO] scale={scale:.4e}, amax={amax:.4f}, "
          f"c_const={c[0]:.3f}, rho={rho}, cap_factor={cap_factor}, "
          f"demand_floor={demand_floor:.3f}")

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(demand, dtype=torch.float32),
        torch.tensor(Starts, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    save_dir = os.path.join(os.getcwd(), "Bilevel Opt Summary")

    model, df = train_bilevel_demand(
        loader=loader,
        gross_data=gross,
        cost_vec=c,
        amax=amax,
        rho=rho,
        epochs=10,
        lr=1e-3,
        save_dir=save_dir,
        cap_factor=cap_factor,
        demand_floor=demand_floor
    )
