import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

data_path = r'C:\Users\19043\Desktop\Thesis\Raw Datasets'
files = ['DJIA.txt', 'NYSE.txt', 'SP500.txt', 'TSE.txt']

# Step1: Read and save raw data without normalization
for file in files:
    file_path = os.path.join(data_path, file)
    data = np.loadtxt(file_path)

    print(f"{file} reading successful, shape: {data.shape}")

    df = pd.DataFrame(data)

    # Save as CSV format (raw values, no normalization)
    csv_name = file.replace('.txt', '_processed.csv')
    csv_path = os.path.join(data_path, csv_name)
    df.to_csv(csv_path, index=False)

    print(f"{file} saved as raw CSV: {csv_name}")

# Step2: Predict with DLinear
# Custom Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=10, pred_len=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Simple DLinear model
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.linear(x)

# Read DJIA first stock (first column, raw values)
df = pd.read_csv(r'C:\Users\19043\Desktop\Thesis\Raw Datasets\DJIA_processed.csv')
data = df.iloc[:, 0].values  # Only use first stock

seq_len = 10
pred_len = 1
dataset = TimeSeriesDataset(data, seq_len, pred_len)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = DLinear(seq_len=seq_len, pred_len=pred_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Simple training
for epoch in range(5):  # 5 epoch examples
    for x, y in loader:
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}')

print("DLinear prediction model's training is complete!")