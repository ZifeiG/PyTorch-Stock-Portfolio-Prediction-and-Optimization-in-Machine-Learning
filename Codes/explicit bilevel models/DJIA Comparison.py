# ================================================================
# Compare KKT vs closed-form u* (no KKT) under tighter (binding) regime
# ================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Model -----------------------------------
class DLinearSimple(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)


# ---------------- Lower-level Solvers ---------------------------
def solve_kkt_lp(d_pred, c, rho, amax, cap_factor):
    """True LP solver (KKT baseline)."""
    n = len(d_pred)
    a = cp.Variable(n, nonneg=True)
    u = cp.Variable(n, nonneg=True)

    cons = [
        cp.sum(a) <= amax,
        u >= d_pred - a,
        a <= cap_factor * d_pred
    ]
    obj = cp.Minimize(c @ a + rho * cp.sum(u))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-7, reltol=1e-7)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)

    if a.value is None:
        return np.zeros(n, dtype=float)
    a_val = np.maximum(np.asarray(a.value).ravel(), 0.0)
    return a_val


def solve_ustar_no_kkt(d_pred, amax, cap_factor):
    """Closed-form approximation (no KKT)."""
    total_d = float(np.sum(d_pred))
    if total_d <= 1e-12:
        return np.zeros_like(d_pred)

    cover = min(amax, total_d)
    a = cover * d_pred / total_d
    a = np.minimum(a, cap_factor * d_pred)
    return a


# ---------------- Training + Evaluation ------------------------
def train_compare(loader, gross_data, cost_vec, amax, rho, epochs, lr, save_dir, cap_factor):
    """
    Train DLinear and compare KKT vs U*.
    Now we also monitor how often the capacity constraint is binding.
    """
    # infer input dim
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
        regrets_kkt, regrets_ustar = [], []
        binding_flags = []  # NEW: track active capacity for oracle a_star

        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            d_pred = torch.sigmoid(model(xb))

            loss = crit(d_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

            with torch.no_grad():
                d_pred_np = d_pred.cpu().numpy()
                d_true_np = yb.cpu().numpy()
                sb_np = sb.cpu().numpy()

                for i in range(d_pred_np.shape[0]):
                    d_p, d_t = d_pred_np[i], d_true_np[i]

                    # predicted allocations
                    a_kkt = solve_kkt_lp(d_p, cost_vec, rho, amax, cap_factor)
                    a_ustar = solve_ustar_no_kkt(d_p, amax, cap_factor)
                    # oracle allocation based on TRUE demand
                    a_star = solve_kkt_lp(d_t, cost_vec, rho, amax, cap_factor)

                    # Evaluate regret (cost-based)
                    u_eval_kkt = np.maximum(d_t - a_kkt, 0.0)
                    u_eval_ustar = np.maximum(d_t - a_ustar, 0.0)
                    u_eval_star = np.maximum(d_t - a_star, 0.0)

                    denom = max(amax, 1e-6)
                    pred_cost_kkt = (cost_vec @ a_kkt + rho * np.sum(u_eval_kkt)) / denom
                    pred_cost_ustar = (cost_vec @ a_ustar + rho * np.sum(u_eval_ustar)) / denom
                    oracle_cost = (cost_vec @ a_star + rho * np.sum(u_eval_star)) / denom

                    regrets_kkt.append(pred_cost_kkt - oracle_cost)
                    regrets_ustar.append(pred_cost_ustar - oracle_cost)

                    # monitor capacity binding for oracle
                    sum_a_star = float(np.sum(a_star))
                    binding_flags.append(sum_a_star >= 0.999 * amax)

                    # Realized return (ex-post)
                    t = int(sb_np[i])
                    if t + 1 < gross_data.shape[0]:
                        next_gross = gross_data[t + 1]
                        w_kkt = a_kkt / max(np.sum(a_kkt), 1e-12)
                        w_ustar = a_ustar / max(np.sum(a_ustar), 1e-12)
                        w_star = a_star / max(np.sum(a_star), 1e-12)
                        realized_kkt = float(next_gross @ w_kkt)
                        realized_ustar = float(next_gross @ w_ustar)
                        oracle_return = float(next_gross @ w_star)
                    else:
                        realized_kkt = realized_ustar = oracle_return = np.nan

                    all_rows.append({
                        "epoch": int(epoch),
                        "index": int(sb_np[i]),
                        "regret_kkt": float(pred_cost_kkt - oracle_cost),
                        "regret_ustar": float(pred_cost_ustar - oracle_cost),
                        "realized_kkt": realized_kkt,
                        "realized_ustar": realized_ustar,
                        "oracle_return": oracle_return,
                        "a_kkt": a_kkt.tolist(),
                        "a_ustar": a_ustar.tolist(),
                        "a_star": a_star.tolist(),
                    })

        avg_bind = float(np.mean(binding_flags)) if len(binding_flags) > 0 else float("nan")
        print(
            f"[EPOCH {epoch+1}/{epochs}] "
            f"loss={epoch_loss/len(loader):.6f} | "
            f"avg_regret(KKT)={np.mean(regrets_kkt):.4f} | "
            f"avg_regret(USTAR)={np.mean(regrets_ustar):.4f} | "
            f"capacity_binding_rate={avg_bind:.2%}"
        )

    df = pd.DataFrame(all_rows)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "bilevel_comparison_final_DJIA.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {out_path}")
    return df


# ------------------------------ Main ---------------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    data_path = os.path.join(base_dir, "Raw Datasets", "train_DJIA.csv")
    print(f"Using dataset: {data_path}")

    gross = pd.read_csv(data_path).values
    T, n = gross.shape

    ret = gross[1:] - gross[:-1]
    pos_ret = np.clip(ret, 0.0, None)

    # Single consistent scale (slightly aggressive but not too small)
    scale = np.percentile(pos_ret, 95) + 1e-8
    demand = np.clip(pos_ret / scale, 0.0, 1.0)

    X = gross[:-1]
    Starts = np.arange(T - 1, dtype=np.int64)

    # --- Tighter parameters to induce binding -------------------
    total_demand = demand.sum(axis=1)

    # Capacity based on a lower quantile so that many samples hit the boundary
    amax = float(np.quantile(total_demand, 0.20))   # 20th percentile
    amax = max(1.0, amax)

    # Per-asset cap, relatively tight
    cap_factor = 0.6

    # Heterogeneous cost vector (NEW): make KKT differ from simple proportional U*
    base_c = 0.20
    vol = pos_ret.std(axis=0) + 1e-8
    vol_norm = vol / vol.mean()
    c = base_c * (1.0 + 0.5 * (vol_norm - 1.0))   # vary ±25%
    c = np.clip(c, 0.05, 0.5)

    # Penalty for unmet demand (keep same across assets)
    rho = 0.6

    print(
        f"[INFO] scale={scale:.4e}, "
        f"amax={amax:.4f}, "
        f"c_mean={c.mean():.3f}, c_min={c.min():.3f}, c_max={c.max():.3f}, "
        f"rho={rho}, cap_factor={cap_factor}"
    )

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(demand, dtype=torch.float32),
        torch.tensor(Starts, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    save_dir = r"C:\Users\19043\Desktop\Thesis\Results and Plots\Bilevel 2\DJIA"
    train_compare(loader, gross, c, amax, rho, epochs=10, lr=1e-3, save_dir=save_dir, cap_factor=cap_factor)