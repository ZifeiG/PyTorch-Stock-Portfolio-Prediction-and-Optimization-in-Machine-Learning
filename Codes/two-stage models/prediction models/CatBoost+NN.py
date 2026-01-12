import pandas as pd
import os
import numpy as np
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load DJIA dataset
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')

#data_path = r"C:\Users\19043\Desktop\Thesis\Raw Datasets\NYSE_processed.csv"
data = pd.read_csv(data_path).values
num_stocks = data.shape[1]

# Parameters
window_size = 60
stride = 20
num_batches = 10

mae_all, mse_all = [], []

for i in range(num_batches):
    start = i * stride
    end = start + window_size
    if end >= len(data):
        break
    X_window = data[start:end]           # shape: (60, 30)
    y_target = data[end]                 # shape: (30,)

    X_train = X_window[:-1]              # t to t+58
    y_train = X_window[1:]               # t+1 to t+59

    y_pred_combined = []

    for stock_idx in range(num_stocks):
        # 1. CatBoost fitting every stock
        model_cb = CatBoostRegressor(verbose=0)
        model_cb.fit(X_train, y_train[:, stock_idx])

        cb_pred = model_cb.predict(X_window[-1].reshape(1, -1))

        # 2. Neural net accepting CatBoost's output as input
        model_nn = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500)
        model_nn.fit(cb_pred.reshape(-1, 1), [y_target[stock_idx]])
        nn_pred = model_nn.predict(cb_pred.reshape(-1, 1))[0]

        y_pred_combined.append(nn_pred)

    # Evaluation
    y_pred_combined = np.array(y_pred_combined)
    mae = mean_absolute_error(y_target, y_pred_combined)
    mse = mean_squared_error(y_target, y_pred_combined)
    mae_all.append(mae)
    mse_all.append(mse)

    print(f"[Batch {i+1}] MAE: {mae:.6f}, MSE: {mse:.6f}")

# Summary
print("\n=== CatBoost + NN Results ===")
print(f"Average MAE: {np.mean(mae_all):.6f}")
print(f"Average MSE: {np.mean(mse_all):.6f}")
