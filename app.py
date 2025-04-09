from flask import Flask, render_template, request, send_file
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Define available companies and indicators
AVAILABLE_COMPANIES = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
                        'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS']
AVAILABLE_INDICATORS = ['SMA', 'EMA', 'ROC', 'ATR', 'OBV']
AVAILABLE_CHART_TYPES = ['line', 'bar', 'scatter', 'histogram']

# LSTM Model Class
class PricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1])

# Preprocess Data for LSTM
def preprocess_data(df):
    prices = df['Close'].values
    mean, std = prices.mean(), prices.std()
    normalized = (prices - mean) / std
    seq_len = 60
    X, y = [], []
    for i in range(len(normalized) - seq_len):
        X.append(normalized[i:i+seq_len])
        y.append(normalized[i+seq_len])
    return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1), torch.tensor(np.array(y), dtype=torch.float32)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_company = None
    selected_indicator = None
    selected_chart_type = None
    candlestick_plot = None
    indicator_plot = None
    prediction_plot = None
    csv_file = None

    if request.method == 'POST':
        selected_company = request.form['company']
        selected_indicator = request.form['indicator']
        selected_chart_type = request.form['chart_type']

        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 1, 1)
        
        ticker = yf.Ticker(selected_company)
        data = ticker.history(start=start_date, end=end_date)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Generate Candlestick Chart
        candlestick_plot = f'static/{selected_company}_candlestick.png'
        mpf.plot(data, type='candle', volume=True, savefig=candlestick_plot)

        # Calculate Indicators
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        data['ROC'] = ((data['Close'] - data['Close'].shift(14)) / data['Close'].shift(14)) * 100
        data['ATR'] = data['High'] - data['Low']
        data['OBV'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()

        # Plot Selected Indicator
        indicator_plot = f'static/{selected_company}_{selected_indicator}.png'
        plt.figure(figsize=(10, 5))
        
        if selected_indicator == 'SMA':
            plt.plot(data.index, data['SMA_50'], label='SMA 50', color='blue')
            plt.plot(data.index, data['SMA_200'], label='SMA 200', color='red')
        elif selected_indicator == 'EMA':
            plt.plot(data.index, data['EMA_50'], label='EMA 50', color='blue')
            plt.plot(data.index, data['EMA_200'], label='EMA 200', color='red')
        elif selected_indicator == 'ROC':
            plt.plot(data.index, data['ROC'], label='Rate of Change (ROC)', color='green')
        elif selected_indicator == 'ATR':
            plt.plot(data.index, data['ATR'], label='Average True Range (ATR)', color='purple')
        elif selected_indicator == 'OBV':
            plt.plot(data.index, data['OBV'], label='On-Balance Volume (OBV)', color='orange')

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'{selected_company} - {selected_indicator} Indicator')
        plt.legend()
        plt.grid()
        plt.savefig(indicator_plot)
        plt.close()

        # Train LSTM Model and Make Predictions
        X, y = preprocess_data(data)
        model = PricePredictor()
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()

        # Generate Future Predictions
        future_days = 30
        predictions = []
        current_seq = X[-1:].clone()
        
        with torch.no_grad():
            for _ in range(future_days):
                pred = model(current_seq)
                predictions.append(pred.item())
                new_pred = pred.reshape(1, 1, 1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_pred], dim=1)
        
        # Convert Predictions Back
        price_mean = data['Close'].mean()
        price_std = data['Close'].std()
        predictions = (np.array(predictions) * price_std) + price_mean

        # Create DataFrame for Predictions
        pred_dates = [end_date + timedelta(days=i) for i in range(1, future_days+1)]
        pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted Price': predictions})
        csv_file = f'static/{selected_company}_predictions.csv'
        pred_df.to_csv(csv_file, index=False)

        # Plot Predictions Based on Selected Chart Type
        prediction_plot = f'static/{selected_company}_predictions.png'
        plt.figure(figsize=(10, 5))
        
        if selected_chart_type == 'line':
            plt.plot(pred_df['Date'], pred_df['Predicted Price'], marker='o', linestyle='-', color='blue', label='Predicted Price')
        elif selected_chart_type == 'bar':
            plt.bar(pred_df['Date'], pred_df['Predicted Price'], color='light blue', label='Predicted Price')
        elif selected_chart_type == 'scatter':
            plt.scatter(pred_df['Date'], pred_df['Predicted Price'], color='green', label='Predicted Price')
        #elif selected_chart_type == 'histogram':
        #    plt.hist(pred_df['Predicted Price'], bins=10, color='blue', alpha=0.7, label='Predicted Price')

        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'{selected_company} - 30-Day Price Prediction ({selected_chart_type.capitalize()})')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()
        plt.savefig(prediction_plot)
        plt.close()

    return render_template('index.html', companies=AVAILABLE_COMPANIES, indicators=AVAILABLE_INDICATORS,
                           chart_types=AVAILABLE_CHART_TYPES, selected_company=selected_company,
                           selected_indicator=selected_indicator, selected_chart_type=selected_chart_type,
                           candlestick_plot=candlestick_plot, indicator_plot=indicator_plot,
                           prediction_plot=prediction_plot, csv_file=csv_file)


@app.route('/about')
def about():
    return render_template('about1.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('main.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f'static/{filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
