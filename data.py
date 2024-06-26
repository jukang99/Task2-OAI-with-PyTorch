
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']]
    return data

def preprocess_data(data, time_step=100, input_size=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(data, time_step):
        X, Y = [], []
        for i in range(len(data)-time_step-1):
            X.append(data[i:(i+time_step), :input_size])
            Y.append(data[i + time_step, :input_size])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], input_size)
    return X, Y, scaler
