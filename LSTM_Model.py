import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def process_prediction(ticker, prediction_days):
    """A function that streamlines the predictive modeling process, allowing external
    applications to only make one call in order to receive price predictions."""
    # Get data that will train LSTM model.
    train_data = get_data(ticker, dt.datetime(2020, 1, 1), (dt.datetime.now() - dt.timedelta(days=30)))

    # Prepare data.
    scaler = MinMaxScaler(feature_range=(0, 1))
    xtrain, ytrain = prepare_data(train_data, scaler, prediction_days)

    # Build and train the LSTM model.
    lstm = build_model(input_shape=(xtrain.shape[1], 1))
    lstm.fit(xtrain, ytrain, epochs=25, batch_size=32)

    # Get data that the LSTM model will test.
    test_data = get_data(ticker, (dt.datetime.now() - dt.timedelta(days=29)), dt.datetime.now())

    # Make predictions on test data.
    total_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
    stock_prophet_prices = []

    for i in range(len(test_data)):
        xtest = total_data[-(prediction_days + len(test_data)) + i:-len(test_data) + i].values.reshape(-1, 1)
        xtest = scaler.transform(xtest)
        xtest = np.array([xtest])
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
        stock_prophet_price = lstm.predict(xtest)
        stock_prophet_prices.append(stock_prophet_price)

    stock_prophet_prices = np.array(stock_prophet_prices).reshape(-1, 1)
    stock_prophet_prices = scaler.inverse_transform(stock_prophet_prices)

    # Plot test predictions.
    real_prices = test_data['Close'].values
    plot_predictions(real_prices, stock_prophet_prices, ticker)

    # Predict the closing price x days from now.
    sp_prediction = stock_prophet_prediction(lstm, scaler, total_data, prediction_days)
    print()
    print()
    print(f'STOCK PROPHET PRICE PREDICTION: {sp_prediction}')
    print()
    input('[HIT ENTER TO CONTINUE]')


def get_data(ticker, start_date, end_date):
    """A function that downloads historical stock data from Yahoo Finance."""
    return yf.download(ticker, start=start_date, end=end_date)


def prepare_data(data, scaler, prediction_days):
    """A function that prepares the stock data for training."""

    # Scale the closing prices.
    scaled_close_prices = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    xtrain = []
    ytrain = []

    for index in range(prediction_days, len(scaled_close_prices)):
        # Create input sequence of length 'prediction_days' for each index.
        xtrain.append(scaled_close_prices[index - prediction_days:index, 0])
        ytrain.append(scaled_close_prices[index, 0])

    # Convert lists to numpy arrays.
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

    # Return prepared input-output pairs
    return xtrain, ytrain


def build_model(input_shape):
    """A function that builds and compiles the LSTM model."""

    # Initialize a Sequential model.
    lstm = Sequential()

    # Add LSTM layers with specified units and return_sequences parameters.
    lstm.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    lstm.add(Dropout(0.2))  # Add dropout layer to prevent over-fitting.
    lstm.add(LSTM(units=50, return_sequences=True))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(units=50))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(units=1))

    # Compile and return the model.
    lstm.compile(optimizer='adam', loss='mean_squared_error')
    return lstm


def plot_predictions(actual_prices, predicted_prices, ticker):
    """A function that plots real price and the Stock Prophet's price."""

    # Plot actual and predicted prices.
    plt.plot(actual_prices, color='black', label=f'Actual Prices')
    plt.plot(predicted_prices, color='purple', label=f'Stock Prophet Prices')
    plt.ylabel('Price')
    plt.xlabel('Last 30 Trading Days')
    plt.title(f'${ticker} Chart')
    plt.legend()

    print()
    print("[CLOSE CHART TO CONTINUE/RECEIVE PRICE PREDICTION]")
    print()

    # Display plot to the user.
    plt.show()


def stock_prophet_prediction(model, scaler, data, prediction_days):
    """A function that predicts the closing price x days from now."""

    # Scale the data.
    lstm_inputs = scaler.transform(data[-prediction_days:].values.reshape(-1, 1))

    # Prepare the input data.
    xtest = np.array([lstm_inputs[-prediction_days:, 0]])
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    # Predict the price using the provided model.
    predicted_price = model.predict(xtest)

    # Inverse transform the predicted price to get the actual price.
    return scaler.inverse_transform(predicted_price)
