import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import date
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import r2_score


class CustomProgressBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, epochs):
        self.progress_bar = progress_bar
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs  # Calculate the progress percentage
        self.progress_bar.progress(progress)  # Update the progress bar


def download_stock_data(stock_symbol, start_date, end_date):  # Download stock data from YF
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data


def plot_adjusted_close(stock_data):  # Plotting Adj Close
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name='Adjusted Close',
                             line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'].rolling(window=100).mean(),
                             mode='lines', name='Trendline', line=dict(color='red', width=0.8)))
    all_time_high_adj_close = stock_data['Adj Close'].idxmax()
    all_time_low_adj_close = stock_data['Adj Close'].idxmin()
    fig.add_trace(go.Scatter(x=[all_time_high_adj_close], y=[stock_data['Adj Close'].loc[all_time_high_adj_close]],
                             mode='markers', marker=dict(color='red', size=10),
                             name='All-Time High Adjusted Closing Price'))
    fig.add_trace(go.Scatter(x=[all_time_low_adj_close], y=[stock_data['Adj Close'].loc[all_time_low_adj_close]],
                             mode='markers', marker=dict(color='green', size=10),
                             name='All-Time Low Adjusted Closing Price'))
    fig.update_layout(annotations=[
        dict(x=all_time_high_adj_close, y=stock_data['Adj Close'].loc[all_time_high_adj_close],
             xref="x", yref="y",
             text="",
             showarrow=True,
             arrowhead=2,
             ax=-30,
             ay=-40),
        dict(x=all_time_low_adj_close, y=stock_data['Adj Close'].loc[all_time_low_adj_close],
             xref="x", yref="y",
             text="",
             showarrow=True,
             arrowhead=2,
             ax=-30,
             ay=40),
    ])
    fig.update_layout(title=f"Adjusted Closing Price Visualization",
                      xaxis_title="Year",
                      yaxis_title="Adjusted Closing Price",
                      showlegend=True)
    return fig


def plot_volume(stock_data):  # Plotting volume
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', opacity=0.5, marker=dict(color='orange')))
    all_time_high_volume = stock_data['Volume'].idxmax()
    all_time_low_volume = stock_data['Volume'].idxmin()
    fig.add_trace(go.Scatter(x=[all_time_high_volume], y=[stock_data['Volume'].loc[all_time_high_volume]],
                             mode='markers', marker=dict(color='red', size=10), name='All-Time High Volume'))
    fig.add_trace(go.Scatter(x=[all_time_low_volume], y=[stock_data['Volume'].loc[all_time_low_volume]],
                             mode='markers', marker=dict(color='green', size=10), name='All-Time Low Volume'))
    fig.update_layout(annotations=[
        dict(x=all_time_high_volume, y=stock_data['Volume'].loc[all_time_high_volume],
             xref="x", yref="y",
             text="",
             showarrow=True,
             arrowhead=2,
             ax=-30,
             ay=-40),
        dict(x=all_time_low_volume, y=stock_data['Volume'].loc[all_time_low_volume],
             xref="x", yref="y",
             text="",
             showarrow=True,
             arrowhead=2,
             ax=-30,
             ay=40),
    ])
    fig.update_layout(title=f"Volume Visualization",
                      xaxis_title="Year",
                      yaxis_title="Volume",
                      showlegend=True)
    return fig


def plot_candlestick(stock_data):  # Plotting Candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Candlestick'))
    all_time_high = stock_data['Close'].idxmax()
    all_time_low = stock_data['Close'].idxmin()
    fig.add_trace(go.Scatter(x=[all_time_high], y=[stock_data['Close'].loc[all_time_high]],
                             mode='markers', marker=dict(color='red', size=10), name='All-Time High'))
    fig.add_trace(go.Scatter(x=[all_time_low], y=[stock_data['Close'].loc[all_time_low]],
                             mode='markers', marker=dict(color='green', size=10), name='All-Time Low'))
    fig.update_layout(annotations=[
        dict(x=all_time_high, y=stock_data['Close'].loc[all_time_high],
             xref="x", yref="y",
             text="",
             showarrow=True,
             arrowhead=2,
             ax=-30,
             ay=-40),
        dict(x=all_time_low, y=stock_data['Close'].loc[all_time_low],
             xref="x", yref="y",
             text="",
             showarrow=True,
             arrowhead=2,
             ax=-30,
             ay=40),
    ])
    fig.update_layout(title=f"Candlestick Chart with RSI and Bollinger Bands",
                      xaxis_title="Year",
                      yaxis_title="Price",
                      showlegend=True)

    return fig


def preprocess_data(data, target_col, lookback):  # Function to preprocess data for LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[target_col].values.reshape(-1, 1))
    x, y = [], []
    for i in range(len(scaled_data) - lookback):
        x.append(scaled_data[i: (i + lookback), 0])
        y.append(scaled_data[i + lookback, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler


def create_lstm_model(input_shape):  # Function to create and train LSTM model
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError(), MeanSquaredError()])
    return model


def create_sequences(data, seq_length):  # Function to create sequences and labels for training the LSTM model
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def evaluate_model(model, X_test, y_test, scaler):  # Function for model evaluation
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test)
    mse = mean_squared_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    return mse, r2


def make_future_predictions(model, current_sequence, scaler, future_steps, sequence_length):
    future_predictions = []
    for _ in range(future_steps):
        current_sequence_reshaped = current_sequence.reshape(1, sequence_length, 1)
        next_prediction = model.predict(current_sequence_reshaped)
        future_predictions.append(next_prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], next_prediction[0, 0])
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions


def plot_results(selected_feature, y_test_original, predictions, future_dates,
                 future_predictions):  # Function to plot results
    fig_results = go.Figure()
    fig_results.add_trace(go.Scatter(x=selected_feature.index[-len(y_test_original):], y=y_test_original.flatten(),
                                     mode='lines', name='True Prices', line=dict(color='blue')))
    fig_results.add_trace(go.Scatter(x=selected_feature.index[-len(predictions):], y=predictions.flatten(),
                                     mode='lines', name='Predicted Prices on Test Set', line=dict(color='red')))
    fig_results.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(),
                                     mode='lines', name='Future Predictions', line=dict(color='green')))
    fig_results.update_layout(title=f'Stock Future Prices Prediction',
                              xaxis_title='Date',
                              yaxis_title='Stock Price',
                              legend=dict(x=0, y=1, traceorder='normal'))
    st.plotly_chart(fig_results, use_container_width=True)


def main():
    # Streamlit Page Layout and Title
    st.set_page_config(layout="wide")
    st.title("LSTM Model: Stock Price Prediction Web App")
    st.info(
        "Some description about the model will be added later. First thing first, go to the left sidebar <------- and choose your desired stock and tune some of the parameters for the LSTM model.")
    st.divider()
    col1, col2 = st.columns(2)

    # Sidebar:
    ## Stock Symbol and Data Range
    st.sidebar.subheader("Data Range Selection & Stock Symbol Input")
    start_date = st.sidebar.date_input("Start Date", date(2013, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2023, 12, 31))
    stock_symbol = st.sidebar.text_input("Stock Symbol", "GOOG")
    st.sidebar.info(
        f"Choose data range of the stock you wish to explore/train. If you need assistance with stock symbols, please head to [Yahoo Finance](https://finance.yahoo.com/)")
    st.sidebar.divider()

    ## Stock Price Prediction: Lookback and future days
    st.sidebar.subheader("Future Price Range Selection & Lookback Duration")
    sequence_length = st.sidebar.number_input("Enter the number of lookback days to predict future prices", min_value=1,
                                              max_value=180, value=30, step=1)
    future_steps = st.sidebar.number_input("Enter the number of future days that you want to predict prices",
                                           min_value=1,
                                           max_value=180, value=7, step=1)
    st.sidebar.info(
        f"Heavy sequence and lookback values may produce more accurate results but the model training will take more than a while.", )
    st.sidebar.divider()

    ## Defining LSTM HyperParameters
    st.sidebar.subheader("Define LSTM Model HyperParameters")
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=50, step=2)

    batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=32, step=2)

    st.sidebar.warning(
        f"If you're finish setting up, don't forget to hit the button below!", )
    trigger_visualization = st.sidebar.button("Explore the Selected Stock")
    # End of Sidebar

    # Start of button action
    if trigger_visualization:
        col1.header("Exploring the Data:")
        with col1.status(f"Downloading data for {stock_symbol} from {start_date} to {end_date}...",
                         expanded=False) as status:
            time.sleep(1)
            stock_data = download_stock_data(stock_symbol, start_date, end_date)
            time.sleep(2)
            status.update(label="Data has been successfully fetched from Yahoo Finance.", state="complete",
                          expanded=False)
        col1.dataframe(stock_data)

        selected_feature = stock_data[['Adj Close']]  # Extracted target column

        fig_adj_closing_price = plot_adjusted_close(stock_data)
        col2.plotly_chart(fig_adj_closing_price, use_container_width=True)
        fig_volume = plot_volume(stock_data)
        col1.plotly_chart(fig_volume, use_container_width=True)
        fig_candlestick = plot_candlestick(stock_data)
        col2.plotly_chart(fig_candlestick, use_container_width=True)
        # End of Basic Data Exploration

        # Normalize the data using Min-Max Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(selected_feature)

        # Create sequences for training
        X, y = create_sequences(data_scaled, sequence_length)

        # Split the data into traininga and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build the LSTM model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape)

        st.divider()
        st.subheader("LSTM Model Training Progress")

        # Train the model with progress bar
        progress_pretext = "Model is training..."
        progress_bar = st.progress(0, text=progress_pretext)  # Initialize the progress bar
        progress_status = st.empty()  # Create an empty element to hold the progress status text

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[CustomProgressBarCallback(progress_bar, epochs)])

        # Update text after training completion
        progress_posttext = "Trained Successfully!"
        progress_status.text(progress_posttext)

        st.subheader("LSTM Model Evaluation Results")

        tab1, tab2, tab3 = st.tabs(["MAE", "MSE", "R2 Score"])

        # Plot MAE
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(x=np.arange(1, len(history.history['mean_absolute_error']) + 1),
                                     y=history.history['mean_absolute_error'],
                                     mode='lines',
                                     name='Training MAE',
                                     line=dict(color='blue')))
        fig_mae.add_trace(go.Scatter(x=np.arange(1, len(history.history['val_mean_absolute_error']) + 1),
                                     y=history.history['val_mean_absolute_error'],
                                     mode='lines',
                                     name='Validation MAE',
                                     line=dict(color='orange')))
        fig_mae.update_layout(title='Training and Validation MAE',
                              xaxis_title='Epochs',
                              yaxis_title='Mean Absolute Error',
                              legend=dict(x=0, y=1, traceorder='normal'))
        with tab1:
            st.plotly_chart(fig_mae, use_container_width=True)

        # Plot MSE
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(x=np.arange(1, len(history.history['mean_squared_error']) + 1),
                                     y=history.history['mean_squared_error'],
                                     mode='lines',
                                     name='Training MSE',
                                     line=dict(color='green')))
        fig_mse.add_trace(go.Scatter(x=np.arange(1, len(history.history['val_mean_squared_error']) + 1),
                                     y=history.history['val_mean_squared_error'],
                                     mode='lines',
                                     name='Validation MSE',
                                     line=dict(color='red')))
        fig_mse.update_layout(title='Training and Validation MSE',
                              xaxis_title='Epochs',
                              yaxis_title='Mean Squared Error',
                              legend=dict(x=0, y=1, traceorder='normal'))
        with tab2:
            st.plotly_chart(fig_mse, use_container_width=True)

        # Evaluate the model
        mse, r2 = evaluate_model(model, X_test, y_test, scaler)

        # Plot R2
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatter(x=[1, 2], y=[r2, r2],
                                    mode='markers',
                                    marker=dict(color=['blue', 'orange']),
                                    text=['Training R2', 'Validation R2']))
        fig_r2.update_layout(title='Training and Validation R2',
                             xaxis_title='Dataset',
                             yaxis_title='R2 Score',
                             showlegend=False)
        with tab3:
            st.plotly_chart(fig_r2, use_container_width=True)

        st.divider()

        # Make future predictions
        current_sequence = X_test[-1]
        future_predictions = make_future_predictions(model, current_sequence, scaler, future_steps, sequence_length)

        # Create future date indices for the predictions
        future_dates = pd.date_range(start=selected_feature.index[-1] + pd.DateOffset(1), periods=future_steps,
                                     freq='D')

        st.subheader("LSTM Future Stock Price Predictions")  # Plot the results
        plot_results(stock_data, scaler.inverse_transform(y_test), scaler.inverse_transform(model.predict(X_test)),
                     future_dates, future_predictions)


if __name__ == "__main__":
    main()
