import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import random
from statsmodels.graphics.tsaplots import plot_acf

#Loading data and cleaning

set1 = '{insert file path here}'
set2 = '{insert file path here}'

df1 = pd.read_parquet(set1, engine='pyarrow')
df2 = pd.read_parquet(set2, engine='pyarrow')

df1 = df1[df1['Date'] != '']
df1 = df1[df1['value'] != 0]
df2 = df2[df2['value'] != 0]

df1['value'] = (df1['value'] != 0).astype(int)
df2['value'] = (df2['value'] != 0).astype(int)

df1['Date'] = pd.to_datetime(df1['Date'], utc=True)
df2['Date'] = pd.to_datetime(df2['Date'], utc=True)

df1.to_csv('df1_cleaned.csv', index=False)
df2.to_csv('df2_cleaned.csv', index=False)

final_df = pd.concat([df1, df2], axis=0, ignore_index=True)

daily_data = final_df.groupby([pd.Grouper(key='Date', freq='D'), 'variable'], observed=True)['value'].size().unstack().fillna(0)
weekly_data = final_df.groupby([pd.Grouper(key='Date', freq='W'), 'variable'], observed=True)['value'].size().unstack().fillna(0)
daily_hyökkäys = daily_data['Hyökkäys']
daily_sota = daily_data['Sota']
daily_pakotteet = daily_data['Pakotteet']
daily_huumeet = daily_data['Huumeet']
daily_tekoäly = daily_data['Tekoäly']


daily_data_rolling = final_df.groupby([pd.Grouper(key='Date', freq='D'), 'variable'], observed=True)['value'].size().unstack().fillna(0).rolling(7, min_periods=1).mean()
daily_hyökkäys_rolling = daily_data_rolling['Hyökkäys']
daily_sota_rolling = daily_data_rolling['Sota']
daily_pakotteet_rolling = daily_data_rolling['Pakotteet']
daily_huumeet_rolling = daily_data_rolling['Huumeet']
daily_tekoäly_rolling = daily_data_rolling['Tekoäly']

# Helper to lock in randomness across 5 seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataset(dataset, window_size, target):
    X, y = [], []
    values = dataset.values
    target_idx = dataset.columns.get_loc(target)
    for i in range(len(values) - window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size, target_idx])
    return np.array(X), np.array(y)

#Cross correlation matrix

correlation_matrix = weekly_data.corr()
plt.figure(figsize=(65, 65))
sns.heatmap(correlation_matrix, cmap='RdYlGn', annot = True, fmt=".2f")
plt.savefig('heatmap')
plt.gcf().autofmt_xdate()
plt.close()

#Cross correlation matrix (used in project) of common terms

sns.set(font_scale=2)
total = weekly_data.sum(axis=0)
common_words = total[total >= 5000].index
weekly_data_common = weekly_data[common_words]
common_correlation_matrix = weekly_data_common.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(common_correlation_matrix, cmap='RdYlGn', annot = True)
plt.savefig('common_heatmap')
plt.gcf().autofmt_xdate()
#plt.close()
plt.show()

#Autocorrelation

plot_acf(daily_hyökkäys.dropna(), lags=28)
plt.show()

#MLR on 4 inputs

lr_scaler = MinMaxScaler()

lr_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = lr_df[lr_df.index >= '2025-01-01'].index

features = ['sota', 'pakotteet', 'huumeet', 'tekoäly']
for col in features:
    lr_df[col] = lr_df[col].shift(1)
lr_df = lr_df.dropna()

train_lr = lr_df[lr_df.index < '2025-01-01']
test_lr  = lr_df[lr_df.index >= '2025-01-01']

train_lr_scaled = lr_scaler.fit_transform(train_lr)
test_lr_scaled = lr_scaler.transform(test_lr)

train_lr = pd.DataFrame(train_lr_scaled, columns=train_lr.columns, index=train_lr.index)
test_lr = pd.DataFrame(test_lr_scaled, columns=test_lr.columns, index=test_lr.index)

range_lr = lr_scaler.data_max_[0] - lr_scaler.data_min_[0]

linear = LinearRegression()
linear.fit(train_lr[features], train_lr['hyökkäys'])

preds_scaled = linear.predict(test_lr[features])

y_test_unscaled_lr = (test_lr['hyökkäys'] * range_lr) + lr_scaler.data_min_[0]
preds_unscaled = (preds_scaled * range_lr) + lr_scaler.data_min_[0]

lr_unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_lr, preds_unscaled))
lr_unscaled_mae = mean_absolute_error(y_test_unscaled_lr, preds_unscaled)

print(f"Linear Regression RMSE: {lr_unscaled_rmse:.2f}")
print(f"Linear Regression MAE: {lr_unscaled_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_lr.values, label='Actual', color='blue')
plt.plot(test_dates, preds_unscaled, label='Predicted', color='red')
plt.title(f'MLR Forecast (RMSE: {lr_unscaled_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

differences = np.abs(y_test_unscaled_lr - preds_unscaled)
plt.figure(figsize=(12, 5))
plt.plot(test_lr.index, differences, color='blue', label='Absolute Error')
plt.title('Residuals (Absolute Differences)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

#MLP on 4 inputs

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(128, 1))

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

MLP_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = MLP_df[MLP_df.index >= '2025-01-01'].index
test_MLP  = MLP_df[MLP_df.index >= '2025-01-01']
train_MLP_unscaled = MLP_df[MLP_df.index < '2025-01-01']

train_end = train_MLP_unscaled.iloc[-window_size:]
test_MLP = pd.concat([train_end, test_MLP])

train_scaled_MLP = scaler.fit_transform(train_MLP_unscaled)
test_scaled_MLP = scaler.transform(test_MLP)

train_MLP = pd.DataFrame(train_scaled_MLP, columns=train_MLP_unscaled.columns, index=train_MLP_unscaled.index)
test_MLP = pd.DataFrame(test_scaled_MLP, columns=test_MLP.columns, index=test_MLP.index)

X_train_MLP, y_train_MLP = create_dataset(train_MLP, window_size, target_col)
X_train_MLP = torch.tensor(X_train_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_MLP = torch.tensor(y_train_MLP, dtype=torch.float32)

X_test_MLP, y_test_MLP = create_dataset(test_MLP, window_size, target_col)
X_test_MLP = torch.tensor(X_test_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_MLP = torch.tensor(y_test_MLP, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_MLP, y_train_MLP), batch_size=128, shuffle=False)

    model = MLPModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025525568759347437, weight_decay=2.7014980830955693e-05)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_MLP = model(X_batch)
            loss = criterion(y_pred_train_MLP.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_MLP.to(device)
        y_test_dev = y_test_MLP.to(device)

        y_pred_test_MLP = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_MLP, y_test_dev)

        X_train_dev = X_train_MLP.to(device)
        y_pred_train_MLP = model(X_train_dev).squeeze()

    y_pred_unscaled_MLP = (y_pred_test_MLP.cpu().numpy() * range1) + min_target
    y_pred_unscaled_MLP_train = (y_pred_train_MLP.cpu().numpy() * range1) + min_target
    y_test_unscaled_MLP = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_MLP, y_pred_unscaled_MLP))
    unscaled_mae = mean_absolute_error(y_test_unscaled_MLP, y_pred_unscaled_MLP)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_MLP, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_MLP, label=f'MLP Prediction', color='red')
plt.title(f'MLP Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(MLP_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_MLP.index, train_MLP_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_MLP.index[window_size:]
y_pred_unscaled_MLP_train = pd.Series(y_pred_unscaled_MLP_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_MLP_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#CNN on 4 inputs

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=7, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.46918841733604943),
            nn.Linear(3328, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

CNN_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = CNN_df[CNN_df.index >= '2025-01-01'].index
test_CNN  = CNN_df[CNN_df.index >= '2025-01-01']
train_CNN_unscaled = CNN_df[CNN_df.index < '2025-01-01']

train_end = train_CNN_unscaled.iloc[-window_size:]
test_CNN = pd.concat([train_end, test_CNN])

train_scaled_CNN = scaler.fit_transform(train_CNN_unscaled)
test_scaled_CNN = scaler.transform(test_CNN)

train_CNN = pd.DataFrame(train_scaled_CNN, columns=train_CNN_unscaled.columns, index=train_CNN_unscaled.index)
test_CNN = pd.DataFrame(test_scaled_CNN, columns=test_CNN.columns, index=test_CNN.index)

X_train_CNN, y_train_CNN = create_dataset(train_CNN, window_size, target_col)
X_train_CNN = torch.tensor(X_train_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_CNN = torch.tensor(y_train_CNN, dtype=torch.float32)

X_test_CNN, y_test_CNN = create_dataset(test_CNN, window_size, target_col)
X_test_CNN = torch.tensor(X_test_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_CNN = torch.tensor(y_test_CNN, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 30

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_CNN, y_train_CNN), batch_size=16, shuffle=False)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.05937365768979703, weight_decay=6.38173731636892e-06)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_CNN = model(X_batch)
            loss = criterion(y_pred_train_CNN.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_CNN.to(device)
        y_test_dev = y_test_CNN.to(device)

        y_pred_test_CNN = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_CNN, y_test_dev)

        X_train_dev = X_train_CNN.to(device)
        y_pred_train_CNN = model(X_train_dev).squeeze()

    y_pred_unscaled_CNN = (y_pred_test_CNN.cpu().numpy() * range1) + min_target
    y_pred_unscaled_CNN_train = (y_pred_train_CNN.cpu().numpy() * range1) + min_target
    y_test_unscaled_CNN = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_CNN, y_pred_unscaled_CNN))
    unscaled_mae = mean_absolute_error(y_test_unscaled_CNN, y_pred_unscaled_CNN)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_CNN, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_CNN, label=f'CNN Prediction', color='red')
plt.title(f'CNN Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(CNN_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_CNN.index, train_CNN_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_CNN.index[window_size:]
y_pred_unscaled_CNN_train = pd.Series(y_pred_unscaled_CNN_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_CNN_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#LSTM on 4 inputs
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4682532924294671,
            bidirectional=False
        )
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        a = self.network(out[:, -1, :])
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

LSTM_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = LSTM_df[LSTM_df.index >= '2025-01-01'].index
test_LSTM  = LSTM_df[LSTM_df.index >= '2025-01-01']
train_LSTM_unscaled = LSTM_df[LSTM_df.index < '2025-01-01']

train_end = train_LSTM_unscaled.iloc[-window_size:]
test_LSTM = pd.concat([train_end, test_LSTM])

train_scaled_LSTM = scaler.fit_transform(train_LSTM_unscaled)
test_scaled_LSTM = scaler.transform(test_LSTM)

train_LSTM = pd.DataFrame(train_scaled_LSTM, columns=train_LSTM_unscaled.columns, index=train_LSTM_unscaled.index)
test_LSTM = pd.DataFrame(test_scaled_LSTM, columns=test_LSTM.columns, index=test_LSTM.index)

X_train_LSTM, y_train_LSTM = create_dataset(train_LSTM, window_size, target_col)
X_train_LSTM = torch.tensor(X_train_LSTM, dtype=torch.float32)[:, :, 1:]
y_train_LSTM = torch.tensor(y_train_LSTM, dtype=torch.float32)

X_test_LSTM, y_test_LSTM = create_dataset(test_LSTM, window_size, target_col)
X_test_LSTM = torch.tensor(X_test_LSTM, dtype=torch.float32)[:, :, 1:]
y_test_LSTM = torch.tensor(y_test_LSTM, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)
   
    train_loader = DataLoader(TensorDataset(X_train_LSTM, y_train_LSTM), batch_size=8, shuffle=False)
   
    model = LSTMModel().to(device)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0036566399318568826, weight_decay=0.0004339864508099267)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_LSTM = model(X_batch)
            loss = criterion(y_pred_train_LSTM.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_LSTM.to(device)
        y_test_dev = y_test_LSTM.to(device)
       
        y_pred_test_LSTM = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_LSTM, y_test_dev)

        X_train_dev = X_train_LSTM.to(device)
        y_pred_train_LSTM = model(X_train_dev).squeeze()

    y_pred_unscaled_LSTM = (y_pred_test_LSTM.cpu().numpy() * range1) + min_target
    y_pred_unscaled_LSTM_train = (y_pred_train_LSTM.cpu().numpy() * range1) + min_target
    y_test_unscaled_LSTM = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM))
    unscaled_mae = mean_absolute_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM)
   
    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)
   
    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_LSTM, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_LSTM, label=f'LSTM Prediction', color='red')
plt.title(f'LSTM Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(LSTM_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_LSTM.index, train_LSTM_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_LSTM.index[window_size:]
y_pred_unscaled_LSTM_train = pd.Series(y_pred_unscaled_LSTM_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_LSTM_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#Plot of residuals for first 4 graphs

differences_lr = np.abs(y_test_unscaled_lr - preds_unscaled)
differences_MLP = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
differences_CNN = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
differences_LSTM = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)

differences_lr = pd.Series(np.abs(y_test_unscaled_lr - preds_unscaled))
differences_MLP = pd.Series(differences_MLP, index=test_dates)
differences_CNN = pd.Series(differences_CNN, index=test_dates)
differences_LSTM = pd.Series(differences_LSTM, index=test_dates)

plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences_lr.rolling(20).mean(), color='orange', label='Absolute Error of MLR')
plt.plot(test_dates, differences_MLP.rolling(20).mean(), color='blue', label='Absolute Error of MLP')
plt.plot(test_dates, differences_CNN.rolling(20).mean(), color='red', label='Absolute Error of CNN')
plt.plot(test_dates, differences_LSTM.rolling(20).mean(), color='green', label='Absolute Error of LSTM')
plt.title('Residuals', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

#MLR on rolling data

lr_scaler = MinMaxScaler()

lr_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys_rolling,
    'sota': daily_sota_rolling,
    'pakotteet': daily_pakotteet_rolling,
    'huumeet': daily_huumeet_rolling,
    'tekoäly': daily_tekoäly_rolling
}).dropna()

test_dates = lr_df[lr_df.index >= '2025-01-01'].index

features = ['sota', 'pakotteet', 'huumeet', 'tekoäly']
for col in features:
    lr_df[col] = lr_df[col].shift(1)
lr_df = lr_df.dropna()

train_lr = lr_df[lr_df.index < '2025-01-01']
test_lr  = lr_df[lr_df.index >= '2025-01-01']

train_lr_scaled = lr_scaler.fit_transform(train_lr)
test_lr_scaled = lr_scaler.transform(test_lr)

train_lr = pd.DataFrame(train_lr_scaled, columns=train_lr.columns, index=train_lr.index)
test_lr = pd.DataFrame(test_lr_scaled, columns=test_lr.columns, index=test_lr.index)

range_lr = lr_scaler.data_max_[0] - lr_scaler.data_min_[0]

linear = LinearRegression()
linear.fit(train_lr[features], train_lr['hyökkäys'])

preds_scaled = linear.predict(test_lr[features])

y_test_unscaled_lr = (test_lr['hyökkäys'] * range_lr) + lr_scaler.data_min_[0]
preds_unscaled = (preds_scaled * range_lr) + lr_scaler.data_min_[0]

lr_unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_lr, preds_unscaled))
lr_unscaled_mae = mean_absolute_error(y_test_unscaled_lr, preds_unscaled)

print(f"Linear Regression RMSE: {lr_unscaled_rmse:.2f}")
print(f"Linear Regression MAE: {lr_unscaled_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_lr.values, label='Actual', color='blue')
plt.plot(test_dates, preds_unscaled, label='Predicted', color='red')
plt.title(f'MLR Forecast (RMSE: {lr_unscaled_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

differences = np.abs(y_test_unscaled_lr - preds_unscaled)
plt.figure(figsize=(12, 5))
plt.plot(test_lr.index, differences, color='blue', label='Absolute Error')
plt.title('Residuals (Absolute Differences)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

#MLP on rolling data, minimising Huber loss

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, 256),
            nn.Tanh(),
            nn.Dropout(0.15344310121787536),
            nn.Linear(256, 8),
            nn.Tanh(),
            nn.Dropout(0.15344310121787536),
            nn.Linear(8, 1))

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = RobustScaler()

MLP_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys_rolling,
    'sota': daily_sota_rolling,
    'pakotteet': daily_pakotteet_rolling,
    'huumeet': daily_huumeet_rolling,
    'tekoäly': daily_tekoäly_rolling
}).dropna()

test_dates = MLP_df[MLP_df.index >= '2025-01-01'].index
test_MLP  = MLP_df[MLP_df.index >= '2025-01-01']
train_MLP_unscaled = MLP_df[MLP_df.index < '2025-01-01']

train_end = train_MLP_unscaled.iloc[-window_size:]
test_MLP = pd.concat([train_end, test_MLP])

train_scaled_MLP = scaler.fit_transform(train_MLP_unscaled)
test_scaled_MLP = scaler.transform(test_MLP)

train_MLP = pd.DataFrame(train_scaled_MLP, columns=train_MLP_unscaled.columns, index=train_MLP_unscaled.index)
test_MLP = pd.DataFrame(test_scaled_MLP, columns=test_MLP.columns, index=test_MLP.index)

X_train_MLP, y_train_MLP = create_dataset(train_MLP, window_size, target_col)
X_train_MLP = torch.tensor(X_train_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_MLP = torch.tensor(y_train_MLP, dtype=torch.float32)

X_test_MLP, y_test_MLP = create_dataset(test_MLP, window_size, target_col)
X_test_MLP = torch.tensor(X_test_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_MLP = torch.tensor(y_test_MLP, dtype=torch.float32)

range1 = scaler.scale_[0]
min_target = scaler.center_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 90

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_MLP, y_train_MLP), batch_size=32, shuffle=False)

    model = MLPModel().to(device)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0045659112830990275, weight_decay=0.0030084349023267315)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_MLP = model(X_batch)
            loss = criterion(y_pred_train_MLP.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_MLP.to(device)
        y_test_dev = y_test_MLP.to(device)

        y_pred_test_MLP = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_MLP, y_test_dev)

        X_train_dev = X_train_MLP.to(device)
        y_pred_train_MLP = model(X_train_dev).squeeze()

    y_pred_unscaled_MLP = (y_pred_test_MLP.cpu().numpy() * range1) + min_target
    y_pred_unscaled_MLP_train = (y_pred_train_MLP.cpu().numpy() * range1) + min_target
    y_test_unscaled_MLP = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_MLP, y_pred_unscaled_MLP))
    unscaled_mae = mean_absolute_error(y_test_unscaled_MLP, y_pred_unscaled_MLP)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_MLP, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_MLP, label=f'MLP Prediction', color='red')
plt.title(f'MLP Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(MLP_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_MLP.index, train_MLP_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_MLP.index[window_size:]
y_pred_unscaled_MLP_train = pd.Series(y_pred_unscaled_MLP_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_MLP_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#CNN on rolling data, minimising RMSE
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=2, padding="valid"),
            nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding="valid"),
            nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding="valid"),
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(0.4312734877166184),
            nn.Linear(80, 1)
        )

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 10
target_col = 'hyökkäys'
scaler = MinMaxScaler()

CNN_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys_rolling,
    'sota': daily_sota_rolling,
    'pakotteet': daily_pakotteet_rolling,
    'huumeet': daily_huumeet_rolling,
    'tekoäly': daily_tekoäly_rolling
}).dropna()

test_dates = CNN_df[CNN_df.index >= '2025-01-01'].index
test_CNN  = CNN_df[CNN_df.index >= '2025-01-01']
train_CNN_unscaled = CNN_df[CNN_df.index < '2025-01-01']

train_end = train_CNN_unscaled.iloc[-window_size:]
test_CNN = pd.concat([train_end, test_CNN])

train_scaled_CNN = scaler.fit_transform(train_CNN_unscaled)
test_scaled_CNN = scaler.transform(test_CNN)

train_CNN = pd.DataFrame(train_scaled_CNN, columns=train_CNN_unscaled.columns, index=train_CNN_unscaled.index)
test_CNN = pd.DataFrame(test_scaled_CNN, columns=test_CNN.columns, index=test_CNN.index)

X_train_CNN, y_train_CNN = create_dataset(train_CNN, window_size, target_col)
X_train_CNN = torch.tensor(X_train_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_CNN = torch.tensor(y_train_CNN, dtype=torch.float32).view(-1, 1)

X_test_CNN, y_test_CNN = create_dataset(test_CNN, window_size, target_col)
X_test_CNN = torch.tensor(X_test_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_CNN = torch.tensor(y_test_CNN, dtype=torch.float32).view(-1, 1)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 120

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_CNN, y_train_CNN), batch_size=8, shuffle=False)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07891875640496554, weight_decay=0.002344073383149448)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_CNN = model(X_batch)
            loss = criterion(y_pred_train_CNN, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_CNN.to(device)
        y_test_dev = y_test_CNN.to(device)

        y_pred_test_CNN = model(X_test_dev)
        test_loss = criterion(y_pred_test_CNN, y_test_dev)

        X_train_dev = X_train_CNN.to(device)
        y_pred_train_CNN = model(X_train_dev)

    y_pred_unscaled_CNN = (y_pred_test_CNN.cpu().numpy() * range1) + min_target
    y_pred_unscaled_CNN_train = (y_pred_train_CNN.cpu().numpy() * range1) + min_target
    y_test_unscaled_CNN = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_CNN, y_pred_unscaled_CNN))
    unscaled_mae = mean_absolute_error(y_test_unscaled_CNN, y_pred_unscaled_CNN)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_CNN, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_CNN, label=f'CNN Prediction', color='red')
plt.title(f'CNN Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(CNN_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_CNN.index, train_CNN_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_CNN.index[window_size:]
y_pred_unscaled_CNN_train = y_pred_unscaled_CNN_train.squeeze()
y_pred_unscaled_CNN_train = pd.Series(y_pred_unscaled_CNN_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_CNN_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#LSTM on rolling data, minimising Huber loss
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=4, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            dropout=0.1075017734559304,
            bidirectional=False
        )
        self.network = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.10302052485566857),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        a = self.network(out[:, -1, :])
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 45
target_col = 'hyökkäys'
scaler = RobustScaler()

LSTM_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys_rolling,
    'sota': daily_sota_rolling,
    'pakotteet': daily_pakotteet_rolling,
    'huumeet': daily_huumeet_rolling,
    'tekoäly': daily_tekoäly_rolling
}).dropna()

test_dates = LSTM_df[LSTM_df.index >= '2025-01-01'].index
test_LSTM  = LSTM_df[LSTM_df.index >= '2025-01-01']
train_LSTM_unscaled = LSTM_df[LSTM_df.index < '2025-01-01']

train_end = train_LSTM_unscaled.iloc[-window_size:]
test_LSTM = pd.concat([train_end, test_LSTM])

train_scaled_LSTM = scaler.fit_transform(train_LSTM_unscaled)
test_scaled_LSTM = scaler.transform(test_LSTM)

train_LSTM = pd.DataFrame(train_scaled_LSTM, columns=train_LSTM_unscaled.columns, index=train_LSTM_unscaled.index)
test_LSTM = pd.DataFrame(test_scaled_LSTM, columns=test_LSTM.columns, index=test_LSTM.index)

X_train_LSTM, y_train_LSTM = create_dataset(train_LSTM, window_size, target_col)
X_train_LSTM = torch.tensor(X_train_LSTM, dtype=torch.float32)[:, :, 1:]
y_train_LSTM = torch.tensor(y_train_LSTM, dtype=torch.float32)

X_test_LSTM, y_test_LSTM = create_dataset(test_LSTM, window_size, target_col)
X_test_LSTM = torch.tensor(X_test_LSTM, dtype=torch.float32)[:, :, 1:]
y_test_LSTM = torch.tensor(y_test_LSTM, dtype=torch.float32)

range1 = scaler.scale_[0]
min_target = scaler.center_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 60

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)
    
    train_loader = DataLoader(TensorDataset(X_train_LSTM, y_train_LSTM), batch_size=128, shuffle=False)
    
    model = LSTMModel().to(device)
    
    criterion = nn.HuberLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.007474507392150614, weight_decay=0.008196919384737864)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_LSTM = model(X_batch)
            loss = criterion(y_pred_train_LSTM.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_LSTM.to(device)
        y_test_dev = y_test_LSTM.to(device)
        
        y_pred_test_LSTM = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_LSTM, y_test_dev)

        X_train_dev = X_train_LSTM.to(device)
        y_pred_train_LSTM = model(X_train_dev).squeeze()

    y_pred_unscaled_LSTM = (y_pred_test_LSTM.cpu().numpy() * range1) + min_target
    y_pred_unscaled_LSTM_train = (y_pred_train_LSTM.cpu().numpy() * range1) + min_target
    y_test_unscaled_LSTM = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM))
    unscaled_mae = mean_absolute_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM)
    
    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)
    
    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_LSTM, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_LSTM, label=f'LSTM Prediction', color='red')
plt.title(f'LSTM Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(LSTM_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_LSTM.index, train_LSTM_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_LSTM.index[window_size:]
y_pred_unscaled_LSTM_train = pd.Series(y_pred_unscaled_LSTM_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_LSTM_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#Plot of residuals for second set of 4 graphs

differences_lr = np.abs(y_test_unscaled_lr - preds_unscaled)
differences_MLP = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
differences_CNN = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN).squeeze()
differences_LSTM = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)

differences_lr = pd.Series(np.abs(y_test_unscaled_lr - preds_unscaled))
differences_MLP = pd.Series(differences_MLP, index=test_dates)
differences_CNN = pd.Series(differences_CNN, index=test_dates)
differences_LSTM = pd.Series(differences_LSTM, index=test_dates)

plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences_lr.rolling(20).mean(), color='orange', label='Absolute Error of MLR')
plt.plot(test_dates, differences_MLP.rolling(20).mean(), color='blue', label='Absolute Error of MLP')
plt.plot(test_dates, differences_CNN.rolling(20).mean(), color='red', label='Absolute Error of CNN')
plt.plot(test_dates, differences_LSTM.rolling(20).mean(), color='green', label='Absolute Error of LSTM')
plt.title('Residuals', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

#MLR on 3 inputs (excl sota)

lr_scaler = MinMaxScaler()

lr_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = lr_df[lr_df.index >= '2025-01-01'].index

features = ['pakotteet', 'huumeet', 'tekoäly']
for col in features:
    lr_df[col] = lr_df[col].shift(1)
lr_df = lr_df.dropna()

train_lr = lr_df[lr_df.index < '2025-01-01']
test_lr  = lr_df[lr_df.index >= '2025-01-01']

train_lr_scaled = lr_scaler.fit_transform(train_lr)
test_lr_scaled = lr_scaler.transform(test_lr)

train_lr = pd.DataFrame(train_lr_scaled, columns=train_lr.columns, index=train_lr.index)
test_lr = pd.DataFrame(test_lr_scaled, columns=test_lr.columns, index=test_lr.index)

range_lr = lr_scaler.data_max_[0] - lr_scaler.data_min_[0]

linear = LinearRegression()
linear.fit(train_lr[features], train_lr['hyökkäys'])

preds_scaled = linear.predict(test_lr[features])

y_test_unscaled_lr = (test_lr['hyökkäys'] * range_lr) + lr_scaler.data_min_[0]
preds_unscaled = (preds_scaled * range_lr) + lr_scaler.data_min_[0]

lr_unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_lr, preds_unscaled))
lr_unscaled_mae = mean_absolute_error(y_test_unscaled_lr, preds_unscaled)

print(f"Linear Regression RMSE: {lr_unscaled_rmse:.2f}")
print(f"Linear Regression MAE: {lr_unscaled_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_lr.values, label='Actual', color='blue')
plt.plot(test_dates, preds_unscaled, label='Predicted', color='red')
plt.title(f'MLR Forecast (RMSE: {lr_unscaled_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

differences = np.abs(y_test_unscaled_lr - preds_unscaled)
plt.figure(figsize=(12, 5))
plt.plot(test_lr.index, differences, color='blue', label='Absolute Error')
plt.title('Residuals (Absolute Differences)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

#MLP on 3 inputs (excl sota)

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(128, 1))

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

MLP_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = MLP_df[MLP_df.index >= '2025-01-01'].index
test_MLP  = MLP_df[MLP_df.index >= '2025-01-01']
train_MLP_unscaled = MLP_df[MLP_df.index < '2025-01-01']

train_end = train_MLP_unscaled.iloc[-window_size:]
test_MLP = pd.concat([train_end, test_MLP])

train_scaled_MLP = scaler.fit_transform(train_MLP_unscaled)
test_scaled_MLP = scaler.transform(test_MLP)

train_MLP = pd.DataFrame(train_scaled_MLP, columns=train_MLP_unscaled.columns, index=train_MLP_unscaled.index)
test_MLP = pd.DataFrame(test_scaled_MLP, columns=test_MLP.columns, index=test_MLP.index)

X_train_MLP, y_train_MLP = create_dataset(train_MLP, window_size, target_col)
X_train_MLP = torch.tensor(X_train_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_MLP = torch.tensor(y_train_MLP, dtype=torch.float32)

X_test_MLP, y_test_MLP = create_dataset(test_MLP, window_size, target_col)
X_test_MLP = torch.tensor(X_test_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_MLP = torch.tensor(y_test_MLP, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_MLP, y_train_MLP), batch_size=128, shuffle=False)

    model = MLPModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025525568759347437, weight_decay=2.7014980830955693e-05)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_MLP = model(X_batch)
            loss = criterion(y_pred_train_MLP.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_MLP.to(device)
        y_test_dev = y_test_MLP.to(device)

        y_pred_test_MLP = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_MLP, y_test_dev)

        X_train_dev = X_train_MLP.to(device)
        y_pred_train_MLP = model(X_train_dev).squeeze()

    y_pred_unscaled_MLP = (y_pred_test_MLP.cpu().numpy() * range1) + min_target
    y_pred_unscaled_MLP_train = (y_pred_train_MLP.cpu().numpy() * range1) + min_target
    y_test_unscaled_MLP = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_MLP, y_pred_unscaled_MLP))
    unscaled_mae = mean_absolute_error(y_test_unscaled_MLP, y_pred_unscaled_MLP)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_MLP, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_MLP, label=f'MLP Prediction', color='red')
plt.title(f'MLP Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(MLP_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_MLP.index, train_MLP_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_MLP.index[window_size:]
y_pred_unscaled_MLP_train = pd.Series(y_pred_unscaled_MLP_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_MLP_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#CNN on 3 inputs (excl sota)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.46918841733604943),
            nn.Linear(3328, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

CNN_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = CNN_df[CNN_df.index >= '2025-01-01'].index
test_CNN  = CNN_df[CNN_df.index >= '2025-01-01']
train_CNN_unscaled = CNN_df[CNN_df.index < '2025-01-01']

train_end = train_CNN_unscaled.iloc[-window_size:]
test_CNN = pd.concat([train_end, test_CNN])

train_scaled_CNN = scaler.fit_transform(train_CNN_unscaled)
test_scaled_CNN = scaler.transform(test_CNN)

train_CNN = pd.DataFrame(train_scaled_CNN, columns=train_CNN_unscaled.columns, index=train_CNN_unscaled.index)
test_CNN = pd.DataFrame(test_scaled_CNN, columns=test_CNN.columns, index=test_CNN.index)

X_train_CNN, y_train_CNN = create_dataset(train_CNN, window_size, target_col)
X_train_CNN = torch.tensor(X_train_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_CNN = torch.tensor(y_train_CNN, dtype=torch.float32)

X_test_CNN, y_test_CNN = create_dataset(test_CNN, window_size, target_col)
X_test_CNN = torch.tensor(X_test_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_CNN = torch.tensor(y_test_CNN, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 30

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_CNN, y_train_CNN), batch_size=16, shuffle=False)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.05937365768979703, weight_decay=6.38173731636892e-06)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_CNN = model(X_batch)
            loss = criterion(y_pred_train_CNN.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_CNN.to(device)
        y_test_dev = y_test_CNN.to(device)

        y_pred_test_CNN = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_CNN, y_test_dev)

        X_train_dev = X_train_CNN.to(device)
        y_pred_train_CNN = model(X_train_dev).squeeze()

    y_pred_unscaled_CNN = (y_pred_test_CNN.cpu().numpy() * range1) + min_target
    y_pred_unscaled_CNN_train = (y_pred_train_CNN.cpu().numpy() * range1) + min_target
    y_test_unscaled_CNN = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_CNN, y_pred_unscaled_CNN))
    unscaled_mae = mean_absolute_error(y_test_unscaled_CNN, y_pred_unscaled_CNN)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_CNN, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_CNN, label=f'CNN Prediction', color='red')
plt.title(f'CNN Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(CNN_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_CNN.index, train_CNN_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_CNN.index[window_size:]
y_pred_unscaled_CNN_train = pd.Series(y_pred_unscaled_CNN_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_CNN_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#LSTM on 3 inputs (excl sota)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4682532924294671,
            bidirectional=False
        )
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        a = self.network(out[:, -1, :])
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

LSTM_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = LSTM_df[LSTM_df.index >= '2025-01-01'].index
test_LSTM  = LSTM_df[LSTM_df.index >= '2025-01-01']
train_LSTM_unscaled = LSTM_df[LSTM_df.index < '2025-01-01']

train_end = train_LSTM_unscaled.iloc[-window_size:]
test_LSTM = pd.concat([train_end, test_LSTM])

train_scaled_LSTM = scaler.fit_transform(train_LSTM_unscaled)
test_scaled_LSTM = scaler.transform(test_LSTM)

train_LSTM = pd.DataFrame(train_scaled_LSTM, columns=train_LSTM_unscaled.columns, index=train_LSTM_unscaled.index)
test_LSTM = pd.DataFrame(test_scaled_LSTM, columns=test_LSTM.columns, index=test_LSTM.index)

X_train_LSTM, y_train_LSTM = create_dataset(train_LSTM, window_size, target_col)
X_train_LSTM = torch.tensor(X_train_LSTM, dtype=torch.float32)[:, :, 1:]
y_train_LSTM = torch.tensor(y_train_LSTM, dtype=torch.float32)

X_test_LSTM, y_test_LSTM = create_dataset(test_LSTM, window_size, target_col)
X_test_LSTM = torch.tensor(X_test_LSTM, dtype=torch.float32)[:, :, 1:]
y_test_LSTM = torch.tensor(y_test_LSTM, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)
   
    train_loader = DataLoader(TensorDataset(X_train_LSTM, y_train_LSTM), batch_size=8, shuffle=False)
   
    model = LSTMModel().to(device)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0036566399318568826, weight_decay=0.0004339864508099267)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_LSTM = model(X_batch)
            loss = criterion(y_pred_train_LSTM.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_LSTM.to(device)
        y_test_dev = y_test_LSTM.to(device)
       
        y_pred_test_LSTM = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_LSTM, y_test_dev)

        X_train_dev = X_train_LSTM.to(device)
        y_pred_train_LSTM = model(X_train_dev).squeeze()

    y_pred_unscaled_LSTM = (y_pred_test_LSTM.cpu().numpy() * range1) + min_target
    y_pred_unscaled_LSTM_train = (y_pred_train_LSTM.cpu().numpy() * range1) + min_target
    y_test_unscaled_LSTM = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM))
    unscaled_mae = mean_absolute_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM)
   
    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)
   
    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_LSTM, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_LSTM, label=f'LSTM Prediction', color='red')
plt.title(f'LSTM Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(LSTM_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_LSTM.index, train_LSTM_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_LSTM.index[window_size:]
y_pred_unscaled_LSTM_train = pd.Series(y_pred_unscaled_LSTM_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_LSTM_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#MLR on 3 inputs (excl pakotteet)

lr_scaler = MinMaxScaler()

lr_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = lr_df[lr_df.index >= '2025-01-01'].index

features = ['sota', 'huumeet', 'tekoäly']
for col in features:
    lr_df[col] = lr_df[col].shift(1)
lr_df = lr_df.dropna()

train_lr = lr_df[lr_df.index < '2025-01-01']
test_lr  = lr_df[lr_df.index >= '2025-01-01']

train_lr_scaled = lr_scaler.fit_transform(train_lr)
test_lr_scaled = lr_scaler.transform(test_lr)

train_lr = pd.DataFrame(train_lr_scaled, columns=train_lr.columns, index=train_lr.index)
test_lr = pd.DataFrame(test_lr_scaled, columns=test_lr.columns, index=test_lr.index)

range_lr = lr_scaler.data_max_[0] - lr_scaler.data_min_[0]

linear = LinearRegression()
linear.fit(train_lr[features], train_lr['hyökkäys'])

preds_scaled = linear.predict(test_lr[features])

y_test_unscaled_lr = (test_lr['hyökkäys'] * range_lr) + lr_scaler.data_min_[0]
preds_unscaled = (preds_scaled * range_lr) + lr_scaler.data_min_[0]

lr_unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_lr, preds_unscaled))
lr_unscaled_mae = mean_absolute_error(y_test_unscaled_lr, preds_unscaled)

print(f"Linear Regression RMSE: {lr_unscaled_rmse:.2f}")
print(f"Linear Regression MAE: {lr_unscaled_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_lr.values, label='Actual', color='blue')
plt.plot(test_dates, preds_unscaled, label='Predicted', color='red')
plt.title(f'MLR Forecast (RMSE: {lr_unscaled_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

differences = np.abs(y_test_unscaled_lr - preds_unscaled)
plt.figure(figsize=(12, 5))
plt.plot(test_lr.index, differences, color='blue', label='Absolute Error')
plt.title('Residuals (Absolute Differences)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

#MLP on 3 inputs (excl pakotteet)

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(128, 1))

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

MLP_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = MLP_df[MLP_df.index >= '2025-01-01'].index
test_MLP  = MLP_df[MLP_df.index >= '2025-01-01']
train_MLP_unscaled = MLP_df[MLP_df.index < '2025-01-01']

train_end = train_MLP_unscaled.iloc[-window_size:]
test_MLP = pd.concat([train_end, test_MLP])

train_scaled_MLP = scaler.fit_transform(train_MLP_unscaled)
test_scaled_MLP = scaler.transform(test_MLP)

train_MLP = pd.DataFrame(train_scaled_MLP, columns=train_MLP_unscaled.columns, index=train_MLP_unscaled.index)
test_MLP = pd.DataFrame(test_scaled_MLP, columns=test_MLP.columns, index=test_MLP.index)

X_train_MLP, y_train_MLP = create_dataset(train_MLP, window_size, target_col)
X_train_MLP = torch.tensor(X_train_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_MLP = torch.tensor(y_train_MLP, dtype=torch.float32)

X_test_MLP, y_test_MLP = create_dataset(test_MLP, window_size, target_col)
X_test_MLP = torch.tensor(X_test_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_MLP = torch.tensor(y_test_MLP, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_MLP, y_train_MLP), batch_size=128, shuffle=False)

    model = MLPModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025525568759347437, weight_decay=2.7014980830955693e-05)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_MLP = model(X_batch)
            loss = criterion(y_pred_train_MLP.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_MLP.to(device)
        y_test_dev = y_test_MLP.to(device)

        y_pred_test_MLP = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_MLP, y_test_dev)

        X_train_dev = X_train_MLP.to(device)
        y_pred_train_MLP = model(X_train_dev).squeeze()

    y_pred_unscaled_MLP = (y_pred_test_MLP.cpu().numpy() * range1) + min_target
    y_pred_unscaled_MLP_train = (y_pred_train_MLP.cpu().numpy() * range1) + min_target
    y_test_unscaled_MLP = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_MLP, y_pred_unscaled_MLP))
    unscaled_mae = mean_absolute_error(y_test_unscaled_MLP, y_pred_unscaled_MLP)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_MLP, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_MLP, label=f'MLP Prediction', color='red')
plt.title(f'MLP Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(MLP_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_MLP.index, train_MLP_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_MLP.index[window_size:]
y_pred_unscaled_MLP_train = pd.Series(y_pred_unscaled_MLP_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_MLP_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#CNN on 3 inputs (excl pakotteet)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.46918841733604943),
            nn.Linear(3328, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

CNN_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = CNN_df[CNN_df.index >= '2025-01-01'].index
test_CNN  = CNN_df[CNN_df.index >= '2025-01-01']
train_CNN_unscaled = CNN_df[CNN_df.index < '2025-01-01']

train_end = train_CNN_unscaled.iloc[-window_size:]
test_CNN = pd.concat([train_end, test_CNN])

train_scaled_CNN = scaler.fit_transform(train_CNN_unscaled)
test_scaled_CNN = scaler.transform(test_CNN)

train_CNN = pd.DataFrame(train_scaled_CNN, columns=train_CNN_unscaled.columns, index=train_CNN_unscaled.index)
test_CNN = pd.DataFrame(test_scaled_CNN, columns=test_CNN.columns, index=test_CNN.index)

X_train_CNN, y_train_CNN = create_dataset(train_CNN, window_size, target_col)
X_train_CNN = torch.tensor(X_train_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_CNN = torch.tensor(y_train_CNN, dtype=torch.float32)

X_test_CNN, y_test_CNN = create_dataset(test_CNN, window_size, target_col)
X_test_CNN = torch.tensor(X_test_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_CNN = torch.tensor(y_test_CNN, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 30

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_CNN, y_train_CNN), batch_size=16, shuffle=False)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.05937365768979703, weight_decay=6.38173731636892e-06)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_CNN = model(X_batch)
            loss = criterion(y_pred_train_CNN.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_CNN.to(device)
        y_test_dev = y_test_CNN.to(device)

        y_pred_test_CNN = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_CNN, y_test_dev)

        X_train_dev = X_train_CNN.to(device)
        y_pred_train_CNN = model(X_train_dev).squeeze()

    y_pred_unscaled_CNN = (y_pred_test_CNN.cpu().numpy() * range1) + min_target
    y_pred_unscaled_CNN_train = (y_pred_train_CNN.cpu().numpy() * range1) + min_target
    y_test_unscaled_CNN = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_CNN, y_pred_unscaled_CNN))
    unscaled_mae = mean_absolute_error(y_test_unscaled_CNN, y_pred_unscaled_CNN)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_CNN, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_CNN, label=f'CNN Prediction', color='red')
plt.title(f'CNN Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(CNN_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_CNN.index, train_CNN_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_CNN.index[window_size:]
y_pred_unscaled_CNN_train = pd.Series(y_pred_unscaled_CNN_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_CNN_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#LSTM on 3 inputs (excl pakotteet)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4682532924294671,
            bidirectional=False
        )
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        a = self.network(out[:, -1, :])
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

LSTM_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = LSTM_df[LSTM_df.index >= '2025-01-01'].index
test_LSTM  = LSTM_df[LSTM_df.index >= '2025-01-01']
train_LSTM_unscaled = LSTM_df[LSTM_df.index < '2025-01-01']

train_end = train_LSTM_unscaled.iloc[-window_size:]
test_LSTM = pd.concat([train_end, test_LSTM])

train_scaled_LSTM = scaler.fit_transform(train_LSTM_unscaled)
test_scaled_LSTM = scaler.transform(test_LSTM)

train_LSTM = pd.DataFrame(train_scaled_LSTM, columns=train_LSTM_unscaled.columns, index=train_LSTM_unscaled.index)
test_LSTM = pd.DataFrame(test_scaled_LSTM, columns=test_LSTM.columns, index=test_LSTM.index)

X_train_LSTM, y_train_LSTM = create_dataset(train_LSTM, window_size, target_col)
X_train_LSTM = torch.tensor(X_train_LSTM, dtype=torch.float32)[:, :, 1:]
y_train_LSTM = torch.tensor(y_train_LSTM, dtype=torch.float32)

X_test_LSTM, y_test_LSTM = create_dataset(test_LSTM, window_size, target_col)
X_test_LSTM = torch.tensor(X_test_LSTM, dtype=torch.float32)[:, :, 1:]
y_test_LSTM = torch.tensor(y_test_LSTM, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)
   
    train_loader = DataLoader(TensorDataset(X_train_LSTM, y_train_LSTM), batch_size=8, shuffle=False)
   
    model = LSTMModel().to(device)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0036566399318568826, weight_decay=0.0004339864508099267)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_LSTM = model(X_batch)
            loss = criterion(y_pred_train_LSTM.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_LSTM.to(device)
        y_test_dev = y_test_LSTM.to(device)
       
        y_pred_test_LSTM = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_LSTM, y_test_dev)

        X_train_dev = X_train_LSTM.to(device)
        y_pred_train_LSTM = model(X_train_dev).squeeze()

    y_pred_unscaled_LSTM = (y_pred_test_LSTM.cpu().numpy() * range1) + min_target
    y_pred_unscaled_LSTM_train = (y_pred_train_LSTM.cpu().numpy() * range1) + min_target
    y_test_unscaled_LSTM = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM))
    unscaled_mae = mean_absolute_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM)
   
    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)
   
    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_LSTM, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_LSTM, label=f'LSTM Prediction', color='red')
plt.title(f'LSTM Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(LSTM_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_LSTM.index, train_LSTM_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_LSTM.index[window_size:]
y_pred_unscaled_LSTM_train = pd.Series(y_pred_unscaled_LSTM_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_LSTM_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#MLR on 3 inputs (excl huumeet)

lr_scaler = MinMaxScaler()

lr_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = lr_df[lr_df.index >= '2025-01-01'].index

features = ['sota', 'pakotteet', 'tekoäly']
for col in features:
    lr_df[col] = lr_df[col].shift(1)
lr_df = lr_df.dropna()

train_lr = lr_df[lr_df.index < '2025-01-01']
test_lr  = lr_df[lr_df.index >= '2025-01-01']

train_lr_scaled = lr_scaler.fit_transform(train_lr)
test_lr_scaled = lr_scaler.transform(test_lr)

train_lr = pd.DataFrame(train_lr_scaled, columns=train_lr.columns, index=train_lr.index)
test_lr = pd.DataFrame(test_lr_scaled, columns=test_lr.columns, index=test_lr.index)

range_lr = lr_scaler.data_max_[0] - lr_scaler.data_min_[0]

linear = LinearRegression()
linear.fit(train_lr[features], train_lr['hyökkäys'])

preds_scaled = linear.predict(test_lr[features])

y_test_unscaled_lr = (test_lr['hyökkäys'] * range_lr) + lr_scaler.data_min_[0]
preds_unscaled = (preds_scaled * range_lr) + lr_scaler.data_min_[0]

lr_unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_lr, preds_unscaled))
lr_unscaled_mae = mean_absolute_error(y_test_unscaled_lr, preds_unscaled)

print(f"Linear Regression RMSE: {lr_unscaled_rmse:.2f}")
print(f"Linear Regression MAE: {lr_unscaled_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_lr.values, label='Actual', color='blue')
plt.plot(test_dates, preds_unscaled, label='Predicted', color='red')
plt.title(f'MLR Forecast (RMSE: {lr_unscaled_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

differences = np.abs(y_test_unscaled_lr - preds_unscaled)
plt.figure(figsize=(12, 5))
plt.plot(test_lr.index, differences, color='blue', label='Absolute Error')
plt.title('Residuals (Absolute Differences)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

#MLP on 3 inputs (excl huumeet)

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(128, 1))

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

MLP_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = MLP_df[MLP_df.index >= '2025-01-01'].index
test_MLP  = MLP_df[MLP_df.index >= '2025-01-01']
train_MLP_unscaled = MLP_df[MLP_df.index < '2025-01-01']

train_end = train_MLP_unscaled.iloc[-window_size:]
test_MLP = pd.concat([train_end, test_MLP])

train_scaled_MLP = scaler.fit_transform(train_MLP_unscaled)
test_scaled_MLP = scaler.transform(test_MLP)

train_MLP = pd.DataFrame(train_scaled_MLP, columns=train_MLP_unscaled.columns, index=train_MLP_unscaled.index)
test_MLP = pd.DataFrame(test_scaled_MLP, columns=test_MLP.columns, index=test_MLP.index)

X_train_MLP, y_train_MLP = create_dataset(train_MLP, window_size, target_col)
X_train_MLP = torch.tensor(X_train_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_MLP = torch.tensor(y_train_MLP, dtype=torch.float32)

X_test_MLP, y_test_MLP = create_dataset(test_MLP, window_size, target_col)
X_test_MLP = torch.tensor(X_test_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_MLP = torch.tensor(y_test_MLP, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_MLP, y_train_MLP), batch_size=128, shuffle=False)

    model = MLPModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025525568759347437, weight_decay=2.7014980830955693e-05)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_MLP = model(X_batch)
            loss = criterion(y_pred_train_MLP.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_MLP.to(device)
        y_test_dev = y_test_MLP.to(device)

        y_pred_test_MLP = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_MLP, y_test_dev)

        X_train_dev = X_train_MLP.to(device)
        y_pred_train_MLP = model(X_train_dev).squeeze()

    y_pred_unscaled_MLP = (y_pred_test_MLP.cpu().numpy() * range1) + min_target
    y_pred_unscaled_MLP_train = (y_pred_train_MLP.cpu().numpy() * range1) + min_target
    y_test_unscaled_MLP = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_MLP, y_pred_unscaled_MLP))
    unscaled_mae = mean_absolute_error(y_test_unscaled_MLP, y_pred_unscaled_MLP)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_MLP, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_MLP, label=f'MLP Prediction', color='red')
plt.title(f'MLP Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(MLP_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_MLP.index, train_MLP_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_MLP.index[window_size:]
y_pred_unscaled_MLP_train = pd.Series(y_pred_unscaled_MLP_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_MLP_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#CNN on 3 inputs (excl huumeet)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.46918841733604943),
            nn.Linear(3328, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

CNN_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = CNN_df[CNN_df.index >= '2025-01-01'].index
test_CNN  = CNN_df[CNN_df.index >= '2025-01-01']
train_CNN_unscaled = CNN_df[CNN_df.index < '2025-01-01']

train_end = train_CNN_unscaled.iloc[-window_size:]
test_CNN = pd.concat([train_end, test_CNN])

train_scaled_CNN = scaler.fit_transform(train_CNN_unscaled)
test_scaled_CNN = scaler.transform(test_CNN)

train_CNN = pd.DataFrame(train_scaled_CNN, columns=train_CNN_unscaled.columns, index=train_CNN_unscaled.index)
test_CNN = pd.DataFrame(test_scaled_CNN, columns=test_CNN.columns, index=test_CNN.index)

X_train_CNN, y_train_CNN = create_dataset(train_CNN, window_size, target_col)
X_train_CNN = torch.tensor(X_train_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_CNN = torch.tensor(y_train_CNN, dtype=torch.float32)

X_test_CNN, y_test_CNN = create_dataset(test_CNN, window_size, target_col)
X_test_CNN = torch.tensor(X_test_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_CNN = torch.tensor(y_test_CNN, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 30

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_CNN, y_train_CNN), batch_size=16, shuffle=False)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.05937365768979703, weight_decay=6.38173731636892e-06)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_CNN = model(X_batch)
            loss = criterion(y_pred_train_CNN.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_CNN.to(device)
        y_test_dev = y_test_CNN.to(device)

        y_pred_test_CNN = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_CNN, y_test_dev)

        X_train_dev = X_train_CNN.to(device)
        y_pred_train_CNN = model(X_train_dev).squeeze()

    y_pred_unscaled_CNN = (y_pred_test_CNN.cpu().numpy() * range1) + min_target
    y_pred_unscaled_CNN_train = (y_pred_train_CNN.cpu().numpy() * range1) + min_target
    y_test_unscaled_CNN = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_CNN, y_pred_unscaled_CNN))
    unscaled_mae = mean_absolute_error(y_test_unscaled_CNN, y_pred_unscaled_CNN)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_CNN, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_CNN, label=f'CNN Prediction', color='red')
plt.title(f'CNN Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(CNN_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_CNN.index, train_CNN_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_CNN.index[window_size:]
y_pred_unscaled_CNN_train = pd.Series(y_pred_unscaled_CNN_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_CNN_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#LSTM on 3 inputs (excl huumeet)

#LSTM on 4 inputs
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4682532924294671,
            bidirectional=False
        )
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        a = self.network(out[:, -1, :])
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

LSTM_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = LSTM_df[LSTM_df.index >= '2025-01-01'].index
test_LSTM  = LSTM_df[LSTM_df.index >= '2025-01-01']
train_LSTM_unscaled = LSTM_df[LSTM_df.index < '2025-01-01']

train_end = train_LSTM_unscaled.iloc[-window_size:]
test_LSTM = pd.concat([train_end, test_LSTM])

train_scaled_LSTM = scaler.fit_transform(train_LSTM_unscaled)
test_scaled_LSTM = scaler.transform(test_LSTM)

train_LSTM = pd.DataFrame(train_scaled_LSTM, columns=train_LSTM_unscaled.columns, index=train_LSTM_unscaled.index)
test_LSTM = pd.DataFrame(test_scaled_LSTM, columns=test_LSTM.columns, index=test_LSTM.index)

X_train_LSTM, y_train_LSTM = create_dataset(train_LSTM, window_size, target_col)
X_train_LSTM = torch.tensor(X_train_LSTM, dtype=torch.float32)[:, :, 1:]
y_train_LSTM = torch.tensor(y_train_LSTM, dtype=torch.float32)

X_test_LSTM, y_test_LSTM = create_dataset(test_LSTM, window_size, target_col)
X_test_LSTM = torch.tensor(X_test_LSTM, dtype=torch.float32)[:, :, 1:]
y_test_LSTM = torch.tensor(y_test_LSTM, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)
   
    train_loader = DataLoader(TensorDataset(X_train_LSTM, y_train_LSTM), batch_size=8, shuffle=False)
   
    model = LSTMModel().to(device)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0036566399318568826, weight_decay=0.0004339864508099267)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_LSTM = model(X_batch)
            loss = criterion(y_pred_train_LSTM.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_LSTM.to(device)
        y_test_dev = y_test_LSTM.to(device)
       
        y_pred_test_LSTM = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_LSTM, y_test_dev)

        X_train_dev = X_train_LSTM.to(device)
        y_pred_train_LSTM = model(X_train_dev).squeeze()

    y_pred_unscaled_LSTM = (y_pred_test_LSTM.cpu().numpy() * range1) + min_target
    y_pred_unscaled_LSTM_train = (y_pred_train_LSTM.cpu().numpy() * range1) + min_target
    y_test_unscaled_LSTM = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM))
    unscaled_mae = mean_absolute_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM)
   
    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)
   
    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_LSTM, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_LSTM, label=f'LSTM Prediction', color='red')
plt.title(f'LSTM Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(LSTM_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_LSTM.index, train_LSTM_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_LSTM.index[window_size:]
y_pred_unscaled_LSTM_train = pd.Series(y_pred_unscaled_LSTM_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_LSTM_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#MLR on 3 inputs (excl tekoäly)

lr_scaler = MinMaxScaler()

lr_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet,
    'tekoäly': daily_tekoäly
}).dropna()

test_dates = lr_df[lr_df.index >= '2025-01-01'].index

features = ['sota', 'pakotteet', 'huumeet']
for col in features:
    lr_df[col] = lr_df[col].shift(1)
lr_df = lr_df.dropna()

train_lr = lr_df[lr_df.index < '2025-01-01']
test_lr  = lr_df[lr_df.index >= '2025-01-01']

train_lr_scaled = lr_scaler.fit_transform(train_lr)
test_lr_scaled = lr_scaler.transform(test_lr)

train_lr = pd.DataFrame(train_lr_scaled, columns=train_lr.columns, index=train_lr.index)
test_lr = pd.DataFrame(test_lr_scaled, columns=test_lr.columns, index=test_lr.index)

range_lr = lr_scaler.data_max_[0] - lr_scaler.data_min_[0]

linear = LinearRegression()
linear.fit(train_lr[features], train_lr['hyökkäys'])

preds_scaled = linear.predict(test_lr[features])

y_test_unscaled_lr = (test_lr['hyökkäys'] * range_lr) + lr_scaler.data_min_[0]
preds_unscaled = (preds_scaled * range_lr) + lr_scaler.data_min_[0]

lr_unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_lr, preds_unscaled))
lr_unscaled_mae = mean_absolute_error(y_test_unscaled_lr, preds_unscaled)

print(f"Linear Regression RMSE: {lr_unscaled_rmse:.2f}")
print(f"Linear Regression MAE: {lr_unscaled_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_lr.values, label='Actual', color='blue')
plt.plot(test_dates, preds_unscaled, label='Predicted', color='red')
plt.title(f'MLR Forecast (RMSE: {lr_unscaled_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.show()

differences = np.abs(y_test_unscaled_lr - preds_unscaled)
plt.figure(figsize=(12, 5))
plt.plot(test_lr.index, differences, color='blue', label='Absolute Error')
plt.title('Residuals (Absolute Differences)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

#MLP on 3 inputs (excl tekoäly)

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.31826565360302583),
            nn.Linear(128, 1))

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

MLP_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet
}).dropna()

test_dates = MLP_df[MLP_df.index >= '2025-01-01'].index
test_MLP  = MLP_df[MLP_df.index >= '2025-01-01']
train_MLP_unscaled = MLP_df[MLP_df.index < '2025-01-01']

train_end = train_MLP_unscaled.iloc[-window_size:]
test_MLP = pd.concat([train_end, test_MLP])

train_scaled_MLP = scaler.fit_transform(train_MLP_unscaled)
test_scaled_MLP = scaler.transform(test_MLP)

train_MLP = pd.DataFrame(train_scaled_MLP, columns=train_MLP_unscaled.columns, index=train_MLP_unscaled.index)
test_MLP = pd.DataFrame(test_scaled_MLP, columns=test_MLP.columns, index=test_MLP.index)

X_train_MLP, y_train_MLP = create_dataset(train_MLP, window_size, target_col)
X_train_MLP = torch.tensor(X_train_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_MLP = torch.tensor(y_train_MLP, dtype=torch.float32)

X_test_MLP, y_test_MLP = create_dataset(test_MLP, window_size, target_col)
X_test_MLP = torch.tensor(X_test_MLP, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_MLP = torch.tensor(y_test_MLP, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_MLP, y_train_MLP), batch_size=128, shuffle=False)

    model = MLPModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025525568759347437, weight_decay=2.7014980830955693e-05)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_MLP = model(X_batch)
            loss = criterion(y_pred_train_MLP.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_MLP.to(device)
        y_test_dev = y_test_MLP.to(device)

        y_pred_test_MLP = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_MLP, y_test_dev)

        X_train_dev = X_train_MLP.to(device)
        y_pred_train_MLP = model(X_train_dev).squeeze()

    y_pred_unscaled_MLP = (y_pred_test_MLP.cpu().numpy() * range1) + min_target
    y_pred_unscaled_MLP_train = (y_pred_train_MLP.cpu().numpy() * range1) + min_target
    y_test_unscaled_MLP = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_MLP, y_pred_unscaled_MLP))
    unscaled_mae = mean_absolute_error(y_test_unscaled_MLP, y_pred_unscaled_MLP)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_MLP, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_MLP, label=f'MLP Prediction', color='red')
plt.title(f'MLP Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_MLP - y_pred_unscaled_MLP)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(MLP_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_MLP.index, train_MLP_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_MLP.index[window_size:]
y_pred_unscaled_MLP_train = pd.Series(y_pred_unscaled_MLP_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_MLP_train.rolling(20).mean(), color='red', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#CNN on 3 inputs (excl tekoäly)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.46918841733604943),
            nn.Linear(3328, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        a = self.network(x)
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

CNN_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet
}).dropna()

test_dates = CNN_df[CNN_df.index >= '2025-01-01'].index
test_CNN  = CNN_df[CNN_df.index >= '2025-01-01']
train_CNN_unscaled = CNN_df[CNN_df.index < '2025-01-01']

train_end = train_CNN_unscaled.iloc[-window_size:]
test_CNN = pd.concat([train_end, test_CNN])

train_scaled_CNN = scaler.fit_transform(train_CNN_unscaled)
test_scaled_CNN = scaler.transform(test_CNN)

train_CNN = pd.DataFrame(train_scaled_CNN, columns=train_CNN_unscaled.columns, index=train_CNN_unscaled.index)
test_CNN = pd.DataFrame(test_scaled_CNN, columns=test_CNN.columns, index=test_CNN.index)

X_train_CNN, y_train_CNN = create_dataset(train_CNN, window_size, target_col)
X_train_CNN = torch.tensor(X_train_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_train_CNN = torch.tensor(y_train_CNN, dtype=torch.float32)

X_test_CNN, y_test_CNN = create_dataset(test_CNN, window_size, target_col)
X_test_CNN = torch.tensor(X_test_CNN, dtype=torch.float32)[:, :, 1:].permute(0, 2, 1)
y_test_CNN = torch.tensor(y_test_CNN, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 30

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)

    train_loader = DataLoader(TensorDataset(X_train_CNN, y_train_CNN), batch_size=16, shuffle=False)

    model = CNNModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.05937365768979703, weight_decay=6.38173731636892e-06)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_CNN = model(X_batch)
            loss = criterion(y_pred_train_CNN.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_CNN.to(device)
        y_test_dev = y_test_CNN.to(device)

        y_pred_test_CNN = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_CNN, y_test_dev)

        X_train_dev = X_train_CNN.to(device)
        y_pred_train_CNN = model(X_train_dev).squeeze()

    y_pred_unscaled_CNN = (y_pred_test_CNN.cpu().numpy() * range1) + min_target
    y_pred_unscaled_CNN_train = (y_pred_train_CNN.cpu().numpy() * range1) + min_target
    y_test_unscaled_CNN = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_CNN, y_pred_unscaled_CNN))
    unscaled_mae = mean_absolute_error(y_test_unscaled_CNN, y_pred_unscaled_CNN)

    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)

    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_CNN, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_CNN, label=f'CNN Prediction', color='red')
plt.title(f'CNN Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_CNN - y_pred_unscaled_CNN)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(CNN_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_CNN.index, train_CNN_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_CNN.index[window_size:]
y_pred_unscaled_CNN_train = pd.Series(y_pred_unscaled_CNN_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_CNN_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#LSTM on 3 inputs (excl tekoäly)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4682532924294671,
            bidirectional=False
        )
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1350608615199711),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        a = self.network(out[:, -1, :])
        return a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 60
target_col = 'hyökkäys'
scaler = MinMaxScaler()

LSTM_df = pd.DataFrame({
    'hyökkäys': daily_hyökkäys,
    'sota': daily_sota,
    'pakotteet': daily_pakotteet,
    'huumeet': daily_huumeet
}).dropna()

test_dates = LSTM_df[LSTM_df.index >= '2025-01-01'].index
test_LSTM  = LSTM_df[LSTM_df.index >= '2025-01-01']
train_LSTM_unscaled = LSTM_df[LSTM_df.index < '2025-01-01']

train_end = train_LSTM_unscaled.iloc[-window_size:]
test_LSTM = pd.concat([train_end, test_LSTM])

train_scaled_LSTM = scaler.fit_transform(train_LSTM_unscaled)
test_scaled_LSTM = scaler.transform(test_LSTM)

train_LSTM = pd.DataFrame(train_scaled_LSTM, columns=train_LSTM_unscaled.columns, index=train_LSTM_unscaled.index)
test_LSTM = pd.DataFrame(test_scaled_LSTM, columns=test_LSTM.columns, index=test_LSTM.index)

X_train_LSTM, y_train_LSTM = create_dataset(train_LSTM, window_size, target_col)
X_train_LSTM = torch.tensor(X_train_LSTM, dtype=torch.float32)[:, :, 1:]
y_train_LSTM = torch.tensor(y_train_LSTM, dtype=torch.float32)

X_test_LSTM, y_test_LSTM = create_dataset(test_LSTM, window_size, target_col)
X_test_LSTM = torch.tensor(X_test_LSTM, dtype=torch.float32)[:, :, 1:]
y_test_LSTM = torch.tensor(y_test_LSTM, dtype=torch.float32)

range1 = scaler.data_max_[0] - scaler.data_min_[0]
min_target = scaler.data_min_[0]

seeds = [42, 123, 456, 789, 1024]
n_epoch = 50

all_rmse = []
all_mae = []

for seed in seeds:
    set_seed(seed)
   
    train_loader = DataLoader(TensorDataset(X_train_LSTM, y_train_LSTM), batch_size=8, shuffle=False)
   
    model = LSTMModel().to(device)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0036566399318568826, weight_decay=0.0004339864508099267)

    for epoch in range(n_epoch):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred_train_LSTM = model(X_batch)
            loss = criterion(y_pred_train_LSTM.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_LSTM.to(device)
        y_test_dev = y_test_LSTM.to(device)
       
        y_pred_test_LSTM = model(X_test_dev).squeeze()
        test_loss = criterion(y_pred_test_LSTM, y_test_dev)

        X_train_dev = X_train_LSTM.to(device)
        y_pred_train_LSTM = model(X_train_dev).squeeze()

    y_pred_unscaled_LSTM = (y_pred_test_LSTM.cpu().numpy() * range1) + min_target
    y_pred_unscaled_LSTM_train = (y_pred_train_LSTM.cpu().numpy() * range1) + min_target
    y_test_unscaled_LSTM = (y_test_dev.cpu().numpy() * range1) + min_target

    unscaled_rmse = np.sqrt(mean_squared_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM))
    unscaled_mae = mean_absolute_error(y_test_unscaled_LSTM, y_pred_unscaled_LSTM)
   
    all_rmse.append(unscaled_rmse)
    all_mae.append(unscaled_mae)
   
    print(f"Seed {seed} = RMSE: {unscaled_rmse:.2f} and MAE: {unscaled_mae:.2f}")

avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)

print(f"\n Final averages over 5 seeds")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE:  {avg_mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_unscaled_LSTM, label='Actual Frequency', color='blue')
plt.plot(test_dates, y_pred_unscaled_LSTM, label=f'LSTM Prediction', color='red')
plt.title(f'LSTM Forecast (Average RMSE: {avg_rmse:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

differences = np.abs(y_test_unscaled_LSTM - y_pred_unscaled_LSTM)
plt.figure(figsize=(12, 5))
plt.plot(test_dates, differences, color='blue', label='Absolute Error')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))

for col, color in zip(LSTM_df.columns, ['black', 'b', 'g', 'm', 'pink']):
    plt.plot(train_LSTM.index, train_LSTM_unscaled[col].rolling(20).mean(), color=color, label=f'{col} (train)')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

train_pred_index = train_LSTM.index[window_size:]
y_pred_unscaled_LSTM_train = pd.Series(y_pred_unscaled_LSTM_train, index=train_pred_index)
plt.plot(train_pred_index, y_pred_unscaled_LSTM_train.rolling(20).mean(), color='orange', label='hyökkäys (predicted)')

plt.title("Performance on train set")
plt.xlabel("Date")
plt.ylabel("Values")
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2025-01-01'))
plt.legend(loc='upper right')
plt.grid(True)
plt.show()