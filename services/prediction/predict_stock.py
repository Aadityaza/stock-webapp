import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import yfinance as yf
from datetime import date, timedelta
import pickle

from services.prediction.plot import *


# Fetch stock data from Yahoo Finance
def fetch_stock_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(start=start_date, end=end_date, actions=False)
    df = df.drop(['Open', 'High', 'Volume', 'Low'], axis=1)
    return df

# Prepare data for LSTM model
def prepare_data(df, interval):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data = df.values
    scaled_data = min_max_scaler.fit_transform(data)

    x_train, y_train = [], []
    for i in range(interval, len(scaled_data)):
        x_train.append(scaled_data[i-interval:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, min_max_scaler

# Predict future stock prices
def predict_future_prices(model, df, days, interval, scaler):
    predicted_prices = []
    last_window = df[-interval:].values
    for _ in range(days):
        scaled_window = scaler.transform(last_window)
        scaled_window = scaled_window.reshape(1, -1, 1)
        predicted_price = model.predict(scaled_window)
        predicted_price = scaler.inverse_transform(predicted_price)[0, 0]

        predicted_prices.append(predicted_price)
        last_window = np.append(last_window[1:], [[predicted_price]], axis=0)

    return predicted_prices

# Split data into training and test sets
def split_data(df, interval, test_size=0.2):
    data = df.values
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size - interval:]

    return train_data, test_data

# Build and train LSTM model
def build_and_train_model(x_train, y_train, epochs=10):
    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=100))
    model.add(Dense(100))
    model.add(Dense(1))

    adam = optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=adam, loss='mse')

    stop = EarlyStopping(monitor='loss', patience=5)
    model.fit(x_train, y_train, batch_size=512, epochs=epochs, shuffle=True, callbacks=[stop])

    return model

# Train and save model
def train_and_save_model(stock_symbol, epochs=100, past_years=10, start_date=None, end_date=None):
    model_path = f'trained_prediction_model/{stock_symbol}_stock_model.h5'
    scaler_path = f'trained_prediction_model/{stock_symbol}_scaler.pkl'

    # Check if model already exists
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        print("Model loaded successfully.")
        return model, scaler, None
    except (OSError, IOError):
        print("Model not found. Training a new model.")

    # Determine start and end dates
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365 * past_years)

    interval = 90
    df = fetch_stock_data(stock_symbol, start_date, end_date)
    train_data, test_data = split_data(df, interval)
    x_train, y_train, scaler = prepare_data(pd.DataFrame(train_data, columns=['Close']), interval)

    model = build_and_train_model(x_train, y_train, epochs)

    # Save the model and scaler
    model.save(model_path)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    train_predictions = model.predict(x_train)
    train_predictions = scaler.inverse_transform(train_predictions).flatten()

    # Plot model performance on training set
    actual_train_prices = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    return model, scaler, plot_model_performance(actual_train_prices, train_predictions)

# Load model and predict future prices
def load_model_and_predict(stock_symbol, prediction_days, past_years=10, start_date=None, end_date=None):
    model, scaler, plot_of_perform = train_and_save_model(stock_symbol, epochs=100, past_years = past_years)

    model_path = f'trained_prediction_model/{stock_symbol}_stock_model.h5'
    scaler_path = f'trained_prediction_model/{stock_symbol}_scaler.pkl'

    # Determine start and end dates
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365 * past_years)

    # Load the saved model and scaler
    model = load_model(model_path)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    df = fetch_stock_data(stock_symbol, start_date, end_date)

    # Predict future prices
    future_predictions = predict_future_prices(model, df, prediction_days, 90, scaler)

    # Generate future dates for plotting
    future_dates = [end_date + timedelta(days=i) for i in range(prediction_days)]

    return future_dates, future_predictions ,plot_of_perform
