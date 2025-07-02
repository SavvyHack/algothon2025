import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
prices = np.loadtxt("prices.txt", dtype=float).T
asset1, asset2 = prices[0], prices[1]

# Compute spread and z-score
def compute_spread_zscore(p1, p2, window=30):
    beta = np.polyfit(p1, p2, 1)[0]
    spread = p2 - beta * p1
    spread_series = pd.Series(spread)
    z = (spread_series - spread_series.rolling(window).mean()) / spread_series.rolling(window).std()
    return spread_series, z, beta

spread, zscore, beta = compute_spread_zscore(asset1, asset2)
df = pd.DataFrame({'spread': spread, 'zscore': zscore})
df['spread_diff'] = df['spread'].diff()
df['zscore_diff'] = df['zscore'].diff()
df['momentum'] = df['spread'].diff(3)
df['volatility'] = df['spread'].rolling(10).std()
df['target_raw'] = df['spread'].shift(-1) - df['spread']

# Only classify large spread changes
threshold = df['target_raw'].rolling(30).std().mean()
df['label'] = 0
df.loc[df['target_raw'] > threshold, 'label'] = 1
df.loc[df['target_raw'] < -threshold, 'label'] = -1
df.dropna(inplace=True)

# Scale features
features = ['spread', 'zscore', 'spread_diff', 'zscore_diff', 'momentum', 'volatility']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Create sequences
sequence_length = 10
X, y = [], []
for i in range(len(df) - sequence_length):
    X.append(df[features].iloc[i:i+sequence_length].values)
    y.append(df['label'].iloc[i+sequence_length])
X, y = np.array(X), np.array(y)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to binary classification (ignore 0 label)
y_train_bin = (y_train + 1) // 2  # -1 => 0, +1 => 1
y_test_bin = (y_test + 1) // 2

# Build LSTM classification model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, len(features))))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_bin, epochs=20, batch_size=32, validation_data=(X_test, y_test_bin), verbose=1)

# Predict probabilities
y_pred_proba = model.predict(X).flatten()
conf_thresh = 0.6
pred_labels = np.where(y_pred_proba > conf_thresh, 1, 0)
positions = np.where(pred_labels == 1, 1, 0)
positions = np.where(y_pred_proba < (1 - conf_thresh), -1, positions)

# Compute PnL
spread_changes = df['target_raw'].values[sequence_length:]
positions = positions[:len(spread_changes)]
pnl = positions * spread_changes
cumulative_pnl = np.cumsum(pnl)
daily_returns = np.diff(cumulative_pnl)

# Performance metrics
def analyze_pnl(pnl):
    returns = np.diff(pnl)
    avg_daily_return = np.mean(returns)
    volatility = np.std(returns)
    sharpe = avg_daily_return / volatility * np.sqrt(252) if volatility != 0 else 0
    max_dd = np.max(np.maximum.accumulate(pnl) - pnl)
    print(f"Total PnL: {pnl[-1]:.2f}")
    print(f"Average Daily Return: {avg_daily_return:.4f}")
    print(f"Volatility: {volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")
    return returns

returns = analyze_pnl(cumulative_pnl)

# Plot cumulative PnL
plt.figure(figsize=(10, 4))
plt.plot(cumulative_pnl)
plt.title("Cumulative PnL (LSTM Classification Strategy)")
plt.xlabel("Time")
plt.ylabel("PnL")
plt.grid(True)
plt.show()

# Rolling return and volatility
roll_win = 50
rolling_avg = pd.Series(returns).rolling(roll_win).mean()
rolling_vol = pd.Series(returns).rolling(roll_win).std()

plt.figure(figsize=(12, 5))
plt.plot(rolling_avg, label="Rolling Avg Return")
plt.plot(rolling_vol, label="Rolling Volatility")
plt.title("Rolling Metrics (50-Day)")
plt.legend()
plt.grid(True)
plt.show()