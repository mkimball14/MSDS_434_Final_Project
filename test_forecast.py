import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

print("Testing Prophet forecasting functionality...")

# Create models directory
os.makedirs("models", exist_ok=True)

# Create synthetic data (30 days of sentiment scores)
base_date = datetime(2023, 1, 1)
dates = [base_date + timedelta(days=i) for i in range(30)]

# Create a sine wave pattern with some noise for sentiment
sentiment = 0.5 * np.sin(np.linspace(0, 3*np.pi, 30)) + 0.2 + 0.1 * np.random.randn(30)
message_counts = 100 + 50 * np.random.rand(30)

# Create dataframe
df = pd.DataFrame({
    'ds': dates,
    'y': sentiment,
    'message_volume': message_counts
})

print(f"Created synthetic data with {len(df)} data points")
print(df.head())

# Train Prophet model
print("Training Prophet model...")
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.add_regressor('message_volume')
model.fit(df)
print("Model training successful!")

# Make forecasts
print("Generating forecasts...")
future = model.make_future_dataframe(periods=10)  # Forecast 10 days
future['message_volume'] = 100  # Set a default message volume
forecast = model.predict(future)
print("Forecast generated successfully!")

# Create and save plots
print("Creating visualizations...")
fig = model.plot(forecast)
plt.title('Test Forecast')
plt.savefig('models/test_forecast.png')
print("Saved forecast plot to models/test_forecast.png")

fig2 = model.plot_components(forecast)
plt.savefig('models/test_components.png')
print("Saved components plot to models/test_components.png")

print("Test completed successfully!") 