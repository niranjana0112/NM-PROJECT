from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
import io
import base64

app = Flask(__name__)

ticker = 'TSLA'
start_date = '2010-01-01'
end_date = '2025-01-01'

def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return data

stock_data = download_stock_data(ticker, start_date, end_date)

data = stock_data[['Close']].dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_data.index[train_size+time_step:], y_test_actual, label="Actual Stock Price", color='blue')
ax.plot(stock_data.index[train_size+time_step:], predicted_stock_price, label="Predicted Stock Price", color='red')
ax.set_title(f'{ticker} Stock Price Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()

img = io.BytesIO()
fig.savefig(img, format='png')
img.seek(0)
plot_url = base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    rmse = math.sqrt(mean_squared_error(y_test_actual, predicted_stock_price))
    return render_template('index.html', plot_url=plot_url, rmse=rmse)

if __name__ == "__main__":
    app.run(debug=True)
