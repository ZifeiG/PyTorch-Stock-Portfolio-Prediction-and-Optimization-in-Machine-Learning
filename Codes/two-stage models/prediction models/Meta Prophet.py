import pandas as pd
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, 'Raw Datasets', 'NYSE_processed.csv')
data = pd.read_csv(data_path) # shape: (T, num_stocks)

# Set window size
window_size = 60
stride = 20
num_batches = 10
mae_list, mse_list = [], []

for batch in range(num_batches):
    start = batch * stride
    end = start + window_size

    for col in data.columns:  # modeling each stock
        series = data[col].iloc[start:end + 1].reset_index(drop=True)

        # Construct Prophet df
        df_prophet = pd.DataFrame({
            "ds": pd.date_range(start='2000-01-01', periods=window_size, freq='D'),
            "y": series[:-1]  # the first 60 days
        })

        model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df_prophet)

        # Predict next day
        future = model.make_future_dataframe(periods=1, freq='D')
        forecast = model.predict(future)
        y_pred = forecast['yhat'].iloc[-1]
        y_true = series.iloc[-1]

        # Evaluation
        mae = mean_absolute_error([y_true], [y_pred])
        mse = mean_squared_error([y_true], [y_pred])
        mae_list.append(mae)
        mse_list.append(mse)

    print(f"[Batch {batch+1}] Avg MAE: {sum(mae_list)/len(mae_list):.6f}, Avg MSE: {sum(mse_list)/len(mse_list):.6f}")

# Final results
print("\n=== Meta Prophet Summary ===")
print("Average MAE:", sum(mae_list)/len(mae_list))
print("Average MSE:", sum(mse_list)/len(mse_list))
