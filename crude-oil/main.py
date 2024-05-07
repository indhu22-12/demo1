# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('e:\study\crude-oil\crude_oil_prices.csv')

# Convert the 'Date' column to datetime format and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the crude oil prices
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Crude Oil Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define the ARIMA model
order = (5, 1, 0) # ARIMA parameters (p, d, q)
model = ARIMA(train, order=order)

# Fit the ARIMA model
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))[0]

# Calculate mean squared error
mse = mean_squared_error(test, predictions)
print('Mean Squared Error:', mse)

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Prices')
plt.plot(test.index, predictions, color='red', linestyle='--', label='Predicted Prices')
plt.title('Crude Oil Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
