import numpy as np
import pandas as pd
import os
from nixtlats import TimeGPT
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your NYSE dataset
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')
data = pd.read_csv(data_path).values

#data_path = r'C:\Users\19043\Desktop\Thesis\Raw Datasets\NYSE_processed.csv'
#data = pd.read_csv(data_path)
num_stocks = data.shape[1]
data_values = data.values

# TimeGPT API Key
API_KEY = "nixak-8fMYdgIcQdm0v5l6LDMimdLGxemLEViHdH2psKmqKCHVeXfnhdz9BUxuFInoXNlMCmcXvnINiIgPwMMb"
model = TimeGPT(api_key=API_KEY)

# Settings
window_size = 60
stride = 20
num_batches = 10

# Result containers
mae_all, mse_all = [], []

for i in range(num_batches):
    start = i * stride
    end = start + window_size
    if end >= len(data_values) - 1:
        break

    X = data_values[start:end]
    y_true = data_values[end]  # the target day

    # Prepare DataFrame for TimeGPT format
    input_df = pd.DataFrame(X, columns=[f'stock{i}' for i in range(num_stocks)])
    input_df['ds'] = pd.date_range(start='2000-01-01', periods=window_size, freq='D')
    input_df = input_df.melt(id_vars='ds', var_name='unique_id', value_name='y')

    # Forecast with TimeGPT
    try:
        forecast_df = model.forecast(df=input_df, h=1, freq='D')
    except Exception as e:
        print(f"[Batch {i+1}] Error during forecast: {e}")
        continue

    forecast_sorted = forecast_df.sort_values(by='unique_id')
    y_pred = forecast_sorted['TimeGPT'].values

    # Evaluation
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae_all.append(mae)
    mse_all.append(mse)

    print(f"[Batch {i+1}] MAE: {mae:.6f}, MSE: {mse:.6f}")

# Final summary
print("\n=== TimeGPT Final Results ===")
if mae_all:
    print(f"Averaged MAE: {np.mean(mae_all):.6f}")
    print(f"Averaged MSE: {np.mean(mse_all):.6f}")
else:
    print("All batches failed.")
