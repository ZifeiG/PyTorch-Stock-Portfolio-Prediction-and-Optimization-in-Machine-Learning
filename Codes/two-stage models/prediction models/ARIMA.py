import numpy as np
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')

#data_path = r'C:\Users\19043\Desktop\Thesis\Raw Datasets\NYSE_processed.csv'
data = pd.read_csv(data_path).values
num_stocks = data.shape[1]

# Parameters
window_size = 60
stride = 20
num_batches = 10

mae_all, mse_all = [], []

for batch in range(num_batches):
    start = batch * stride
    end = start + window_size
    if end >= len(data): break
    target_day = data[end]

    y_preds = []
    for stock_idx in range(num_stocks):
        series = data[start:end, stock_idx]
        try:
            model = ARIMA(series, order=(5, 1, 0))  # Tuned ARIMA(p,d,q)
            fitted = model.fit()
            forecast = fitted.forecast()[0]
        except:
            forecast = series[-1]  # fallback if model fails
        y_preds.append(forecast)

    mae = mean_absolute_error(target_day, y_preds)
    mse = mean_squared_error(target_day, y_preds)
    mae_all.append(mae)
    mse_all.append(mse)
    print(f"[Batch {batch+1}] MAE: {mae:.6f}, MSE: {mse:.6f}")

print("\n=== Tuned ARIMA Results ===")
print(f"Average MAE: {np.mean(mae_all):.6f}")
print(f"Average MSE: {np.mean(mse_all):.6f}")
